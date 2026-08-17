# -*- coding: utf-8 -*-
"""Legacy static predictive-coding classifier model.

Implement the three-level hierarchical Rao & Ballard-style generative model
with a softmax classification layer, including representation (r) and weight
(U) update math, reconstruction, receptive-field extraction, and weight I/O.
This is part of the validated legacy reference implementation of the
Li/Rogers static PCC model; the extensible models in pypredcoding/ are
numerically parity-tested against it.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import os


class Model:
    """Three-level static PCC model with a softmax classification level."""

    def __init__(self, dataset: Any, iteration: int = 30, prior: str = "kurtotic") -> None:
        """Set default hyperparameters and randomly initialize U1/U2/U3 from the dataset geometry.

        Args:
            dataset: Dataset instance providing RF1 geometry and image count.
        """
        self.dataset = dataset
        self.iteration = iteration
        self.prior = prior # "kurtotic" or "gaussian"
        self.activation = "linear"

        # NOTE: k_r (k1) and k_U (k2) do not match with that from the original paper
        self.k_r = 0.0005 # Learning rate for r
        self.k_U_init = 0.005  # Initial learning rate for U
        # Decay cycle is in units of training inputs; rate is the division denominator.
        self.k_U_decay_cycle = None
        self.k_U_decay_rate = None

        self.sigma_sq0 = 1.0  # Variance of observation distribution of I
        self.sigma_sq1 = 10.0 # Variance of observation distribution of r1
        self.sigma_sq2 = 10.0 # Variance of observation distribution of r2
        self.sigma_sq3 = 2.0 # Variance of observation distribution of r3
        self.alpha1 = 1.0  # Precision param of r1 prior
        self.alpha2 = 0.05 # Precision param of r2 prior
        self.alpha3 = 0.05 # Precision param of r3 prior

        # NOTE: the original paper only provides one lambda value (which is lambda1 here)
        self.lambda1 = 0.02 # Precision param of U1 prior    (var=50.0, std=7.1)
        self.lambda2 = 0.00001 # Precision param of U2 prior
        self.lambda3 = 0.00001

        # Multiplicative scale applied to the initial random U weights.
        U_scale = 1.0
        self.U_scale = U_scale

        # inputs: three 16 x 16 overlapping rf1 patches (offset by 5 pixels horizontally)
        # Level 1: 3 modules, each module has 32 input-estimating neurons and 32 error-detecting neurons
        # Level 2: 128 input-estimating neurons and 128 error-detecting neurons
        self.input_x = dataset.rf1_size[1]
        self.input_y = dataset.rf1_size[0]
        self.input_size = self.input_x * self.input_y
        self.input_offset_x = dataset.rf1_offset_x
        self.input_offset_y = dataset.rf1_offset_y

        self.level1_layout_x = dataset.rf1_layout_size[1]
        self.level1_layout_y = dataset.rf1_layout_size[0]
        self.level1_module_n = self.level1_layout_x * self.level1_layout_y
        self.level1_module_size = 32

        self.level2_module_size = 128

        # U1: level-1 top-down weights
        # 3 modules (one for each rf1 patch); 256 pixels (16 x 16) in each rf1 patch; 32 input-estimating neurons in each module
        self.U1 = (np.random.rand(self.level1_module_n, self.input_size, self.level1_module_size) - 0.5) * self.U_scale
        # U2: level-2 top-down weights
        # 96 (3 x 32) level-1 error-detecting neurons; 128 level-2 input-estimating neurons
        self.U2 = (np.random.rand(self.level1_module_n * self.level1_module_size, self.level2_module_size) - 0.5) * self.U_scale

        self.k_U = self.k_U_init

        # Preserve the older tanh path's explicit level-2 scaling knob.
        self.level2_lr_scale = 1.0

        # level 3 for classification (level-3 consists of localist nodes, one for each training image)
        self.level3_module_size = len(dataset.images)
        self.U3 = (np.random.rand(self.level2_module_size, self.level3_module_size) - 0.5) * self.U_scale

    def initialize_weights(self, weight_init: tuple[str, float] | list) -> None:
        """Re-initialize U1/U2/U3 using a ('uniform', shift) or ('gaussian', std) spec."""
        if not isinstance(weight_init, (tuple, list)) or len(weight_init) != 2:
            raise ValueError(
                "WEIGHT_INIT must be a 2-item tuple/list like ('uniform', -0.5) or ('gaussian', 0.01)."
            )

        mode = str(weight_init[0]).strip().lower()
        value = float(weight_init[1])

        if mode == "uniform":
            self.U1 = (np.random.rand(*self.U1.shape) + value) * self.U_scale
            self.U2 = (np.random.rand(*self.U2.shape) + value) * self.U_scale
            self.U3 = (np.random.rand(*self.U3.shape) + value) * self.U_scale
        elif mode == "gaussian":
            self.U1 = np.random.normal(loc=0.0, scale=value, size=self.U1.shape) * self.U_scale
            self.U2 = np.random.normal(loc=0.0, scale=value, size=self.U2.shape) * self.U_scale
            self.U3 = np.random.normal(loc=0.0, scale=value, size=self.U3.shape) * self.U_scale
        else:
            raise ValueError(
                f"Unsupported WEIGHT_INIT mode '{mode}'. Expected 'uniform' or 'gaussian'."
            )

    def prior_trans(self, x: np.ndarray, prior: str) -> np.ndarray | int:
        """Return the prior-gradient denominator: 1 + x^2 for kurtotic, 1 for gaussian."""
        if prior == "kurtotic":
            x_trans = 1 + np.square(x)
        elif prior == "gaussian":
            x_trans = 1
        else:
            raise ValueError(f"Unsupported prior '{prior}'. Expected 'kurtotic' or 'gaussian'.")
        return x_trans

    def _decay_k_u(self, input_index: int) -> None:
        """Divide k_U by the decay rate when input_index hits a decay-cycle boundary."""
        if self.k_U_decay_cycle is None or self.k_U_decay_rate is None:
            return
        if self.k_U_decay_cycle <= 0:
            raise ValueError("k_U_decay_cycle must be positive when decay is enabled.")
        if self.k_U_decay_rate == 0:
            raise ValueError("k_U_decay_rate must be non-zero when decay is enabled.")
        if input_index % self.k_U_decay_cycle == 0:
            self.k_U = self.k_U / self.k_U_decay_rate

    def apply_input(self, inputs: Sequence[np.ndarray], label: np.ndarray,
                    training: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the linear-activation r/U update loop for one set of RF1 patches.

        Args:
            inputs: RF1 patches for one RF2 patch, one flattened array per module.
            label: One-hot image-identity label; zeroed internally when not training.
            training: When True, also update U1/U2/U3 in place.

        Returns:
            (r1, r2, r3, e1, e2, e3) with level-1 arrays flattened to vectors.
        """
        inputs = np.array(inputs)
        r1 = np.zeros((self.level1_module_n, self.level1_module_size), dtype=np.float32)
        r2 = np.zeros(self.level2_module_size, dtype=np.float32)
        r3 = np.zeros(self.level3_module_size, dtype=np.float32)

        for i in range(self.iteration):
            # predictions
            r10 = np.matmul(self.U1, r1[:, :, None]).squeeze()
            r21 = self.U2.dot(r2).reshape(r1.shape)
            r32 = self.U3.dot(r3)
            r43 = label if training else label*0

            # prediction errors
            e0 = inputs - r10
            e1 = r1 - r21
            e2 = r2 - r32
            e3 = (np.exp(r3)/np.sum(np.exp(r3))) - r43 # softmax cross-entropy loss

            # r updates
            dr1 = (self.k_r/self.sigma_sq0) * np.matmul(np.transpose(self.U1, axes=(0,2,1)), e0[:, :, None]).squeeze() \
                  + (self.k_r/self.sigma_sq1) * -e1 \
                  - self.k_r * self.alpha1 * r1 / self.prior_trans(r1, self.prior)

            dr2 = (self.k_r / self.sigma_sq1) * self.U2.T.dot(e1.flatten()) \
                  + (self.k_r / self.sigma_sq2) * -e2 \
                  - self.k_r * self.alpha2 * r2 / self.prior_trans(r2, self.prior)

            dr3 = (self.k_r / self.sigma_sq2) * self.U3.T.dot(e2) \
                + (self.k_r / self.sigma_sq3) * -e3 \
                  - self.k_r * self.alpha3 * r3 / self.prior_trans(r3, self.prior)

            # U updates
            if training:
                dU1 = (self.k_U/self.sigma_sq0) * np.matmul(e0[:, :, None], r1[:, None, :]) \
                       - self.k_U * self.lambda1 * self.U1 / self.prior_trans(self.U1, self.prior)

                dU2 = (self.k_U / self.sigma_sq1) * np.outer(e1.flatten(), r2) \
                      - self.k_U * self.lambda2 * self.U2 / self.prior_trans(self.U2, self.prior)

                dU3 = (self.k_U / self.sigma_sq2) * np.outer(e2, r3) \
                      - self.k_U * self.lambda3 * self.U3 / self.prior_trans(self.U3, self.prior)

            # apply r updates
            r1 += dr1
            r2 += dr2
            r3 += dr3

            # apply U updates
            if training:
                self.U1 += dU1
                self.U2 += dU2
                self.U3 += dU3

        # flatten level 1 nodes to vectors
        r1 = r1.flatten()
        e1 = e1.flatten()

        return r1, r2, r3, e1, e2, e3

    def apply_input_tanh(self, inputs: Sequence[np.ndarray], label: np.ndarray,
                         training: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the tanh-activation r/U update loop for one set of RF1 patches.

        Mirrors apply_input but with tanh nonlinearities, per-module level-1
        processing, and the level2_lr_scale factor on level-2 updates.

        Returns:
            (r1, r2, r3, e1, e2, e3) with level-1 arrays as flat vectors.
        """
        # 96 (3 x 32) level-1 input-estimating neurons' representations
        r1 = np.zeros([self.level1_module_n * self.level1_module_size], dtype=np.float32)
        # 128 level-2 input-estimating neurons' representations
        r2 = np.zeros([self.level2_module_size], dtype=np.float32)
        # level-3 representations
        r3 = np.zeros([self.level3_module_size], dtype=np.float32)

        # e1 (r1-r21): level 2's bottom-up prediction error (from level 1 to level 2)
        e1 = np.zeros([self.level1_module_n * self.level1_module_size], dtype=np.float32)

        for _ in range(self.iteration):
            # Calculate r21 (level 2's prediction for level 1)
            r21 = np.tanh(self.U2.dot(r2))
            F2 = np.diag(1 - np.square(r21))

            for m in range(self.level1_module_n):
                m_start = self.level1_module_size * m
                m_end = self.level1_module_size * (m + 1)
                I = inputs[m]
                r1_m = r1[m_start:m_end]
                r21_m = r21[m_start:m_end]

                U1_m = self.U1[m]
                r10_m = np.tanh(U1_m.dot(r1_m))
                F1_m = np.diag(1 - np.square(r10_m))

                e0_m = I - r10_m
                e1_m = r1_m - r21_m

                dr1_m = (self.k_r / self.sigma_sq0) * U1_m.T.dot(F1_m.dot(e0_m)) \
                     + (self.k_r / self.sigma_sq1) * -e1_m \
                     - self.k_r * self.alpha1 * r1_m

                if training:
                    dU1_m = (self.k_U / self.sigma_sq0) * np.outer(F1_m.dot(e0_m), r1_m) \
                         - self.k_U * self.lambda1 * U1_m
                    self.U1[m] += dU1_m

                r1[m_start:m_end] += dr1_m
                e1[m_start:m_end] = e1_m

            r32 = np.tanh(self.U3.dot(r3))
            F3 = np.diag(1 - np.square(r32))
            e2 = r2 - r32

            dr2 = (self.k_r * self.level2_lr_scale / self.sigma_sq1) * self.U2.T.dot(F2.dot(e1)) \
                  + (self.k_r * self.level2_lr_scale / self.sigma_sq2) * -e2 \
                  - self.k_r * self.level2_lr_scale * self.alpha2 * r2

            if training:
                dU2 = (self.k_U * self.level2_lr_scale / self.sigma_sq1) * np.outer(F2.dot(e1), r2) \
                      - self.k_U * self.level2_lr_scale * self.lambda2 * self.U2
                self.U2 += dU2

            r2 += dr2

            if not training:
                label = label * 0

            e3 = (np.exp(r3) / np.sum(np.exp(r3))) - label

            dr3 = (self.k_r / self.sigma_sq2) * self.U3.T.dot(F3.dot(e2)) \
                + (self.k_r / self.sigma_sq3) * -e3 \
                  - self.k_r * self.alpha3 * r3

            if training:
                dU3 = (self.k_U / self.sigma_sq2) * np.outer(F3.dot(e2), r3) \
                      - self.k_U * self.lambda3 * self.U3
                self.U3 += dU3

            r3 += dr3

        return r1, r2, r3, e1, e2, e3

    def train(self, dataset: Any, input_indices: Iterable[int] | None = None) -> None:
        """Train on the given dataset's RF2 patches for one epoch.

        Resets k_U to k_U_init at the start, then applies the decay schedule
        after each processed input index.

        Args:
            input_indices: Optional subset of RF2 patch indices; defaults to all.
        """
        if input_indices is None:
            input_indices = range(len(dataset.rf2_patches))
        else:
            input_indices = tuple(int(index) for index in input_indices)

        self.k_U = self.k_U_init

        # One training pass over the selected rf2 patches.
        for i in input_indices:
            rf1_patches = dataset.get_rf1_patches(i)
            label = dataset.labels[i]
            r1, r2, r3, e1, e2, e3 = self.apply_input(rf1_patches, label, training=True)
            self._decay_k_u(i)

        print("train finished")

    def reconstruct(self, r: np.ndarray, level: int = 1) -> np.ndarray:
        """Reconstruct an RF2-sized image from a representation at the given level.

        Projects r down through the U weights; unlike receptive fields, these
        reconstructions depend on the current input's representations.
        """
        if level==1:
            r1 = r # (96,)
        elif level==2:
            r2 = r # (128,)
            r1 = self.U2.dot(r2) # (96,)
        elif level==3:
            r3 = r
            r2 = self.U3.dot(r3)
            r1 = self.U2.dot(r2)

        # reconstructed image size is 16 x 26 because the each set of inputs is three overlapping (offset by 5 pixels horizontally) 16 x 16 rf1 patches
        rf2_patch = np.zeros((self.input_y + (self.input_offset_y * (self.level1_layout_y - 1)), \
                              self.input_x + (self.input_offset_x * (self.level1_layout_x - 1))), dtype=np.float32)

        # Reconstruct each rf1 patch separately, then sum overlapping regions.
        for i in range(self.level1_module_n):
            module_y = i % self.level1_layout_y
            module_x = i // self.level1_layout_y

            r = r1[self.level1_module_size * i:self.level1_module_size * (i+1)]
            U = self.U1[i]
            Ur = U.dot(r).reshape(self.input_y, self.input_x)
            rf2_patch[self.input_offset_y * module_y :self.input_offset_y * module_y + self.input_y, \
                      self.input_offset_x * module_x :self.input_offset_x * module_x + self.input_x] += Ur
        return rf2_patch

    def get_level2_rf(self, index: int) -> np.ndarray:
        """Return the level-2 receptive field for one level-2 neuron.

        Projects the neuron's U2 column through U1; unlike reconstructions,
        receptive fields do not change with input when the model is frozen.
        """
        rf = np.zeros((self.input_y + (self.input_offset_y * (self.level1_layout_y - 1)), \
                       self.input_x + (self.input_offset_x * (self.level1_layout_x - 1))), dtype=np.float32)

        for i in range(self.level1_module_n):
            module_y = i % self.level1_layout_y
            module_x = i // self.level1_layout_y

            U2 = self.U2[:,index][self.level1_module_size * i:self.level1_module_size * (i+1)]
            UU = self.U1[i].dot(U2).reshape((self.input_y, self.input_x))
            rf[self.input_offset_y * module_y :self.input_offset_y * module_y + self.input_y, \
               self.input_offset_x * module_x :self.input_offset_x * module_x + self.input_x] += UU

        return rf

    def save(self, dir_name: str) -> None:
        """Save U1/U2/U3 to a compressed .npz file under dir_name."""
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model")

        np.savez_compressed(file_path,
                            U1=self.U1,
                            U2=self.U2,
                            U3=self.U3)
        print("saved: {}".format(dir_name))

    def load(self, dir_name: str) -> None:
        """Load U1/U2/U3 from dir_name/model.npz; print and return if missing."""
        file_path = os.path.join(dir_name, "model.npz")
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        data = np.load(file_path)
        self.U1 = data["U1"]
        self.U2 = data["U2"]
        self.U3 = data["U3"]
        print("loaded: {}".format(dir_name))
