"""Cost functions for static and recurrent predictive-coding classifiers.

Implements Rao & Ballard (1999)-style representation, prior, and
classification cost terms plus the gradient update rules used by the
StaticPCC (sPCC) and RecurrentPCC (rPCC) models (Rogers 2026 extensions).
Each cost-function object copies the model's attribute dict at construction
so cost and update math reads model state directly.
"""
from __future__ import annotations

import numpy as np
from copy import copy
from functools import partial
from typing import Any, Callable, Iterable


class StaticCostFunction():
    """Cost and gradient-update machinery for a StaticPCC model.

    Copies the model's ``__dict__`` at construction so all cost and update
    methods read and write model state (r, U, hyperparameters) directly.
    Architecture-specific tensor-product helpers are bound by
    ``initialize_components``; method-name suffixes encode the variant:
    ``_fhl`` (flat_hidden_lyrs), ``_e1l`` (expand_first_lyr), ``_e1l_Li``
    (expand_first_lyr with the Li-style 'reshaped' first-to-second-layer
    connection).
    """

    def __init__(self, sPCC: Any) -> None:
        """Copy model attributes and bind architecture-specific components.

        Args:
            sPCC: StaticPCC model instance whose ``__dict__`` is copied.
        """
        # Copy attributes from the model
        self.__dict__.update(sPCC.__dict__)
        
        # Initialize component functions based on architecture and config controls.
        self.initialize_components(self.architecture)
        
        self.rn_topdown_upd_dict = {'c1': self.rn_topdown_upd_c1,
                                    'c2': self.rn_topdown_upd_c2,
                                    None: self.rn_topdown_upd_None}
    
    def initialize_components(self, architecture_key: str) -> None:
        """Bind architecture-specific tensor-product helper implementations.

        Selects 'flat_hidden_lyrs' (fhl) or 'expand_first_lyr' (e1l) variants,
        using '_Li' implementations when the first-to-second-layer connection
        is 'reshaped'.
        """
        if architecture_key == 'flat_hidden_lyrs':
            self.U1mat_mult_r1vecormat = self.U1r1_fhl
            self.U1Tmat_mult_Idiffmat = self.U1T_Idiff_fhl
            self.U2mat_mult_r2vec = self.U2r2_fhl
            self.U2Tmat_mult_L1diffvecormat = self.U2T_L1diff_fhl
            self.U_gteq3_mat_mult_r_gteq3_vec = self.U3r3_fhl
            self.U3Tmat_mult_L2diffvec = self.U3T_L2diff_fhl
            self.Idiffmat_mult_r1vecormat = self.Idiff_r1_fhl
            self.L1diffvecormat_mult_r2vec = self.L1diff_r2_fhl
            self.L_gteq2_diffvec_mult_r_gteq3_vec = self.L2diff_r3_fhl
            return

        if architecture_key != 'expand_first_lyr':
            raise ValueError(f'Unsupported architecture: {architecture_key}')

        li_ops = self.first_to_second_lyr_connection_architecture == 'reshaped'

        self.U1mat_mult_r1vecormat = self.U1r1_e1l_Li if li_ops else self.U1r1_e1l
        self.U1Tmat_mult_Idiffmat = self.U1T_Idiff_e1l_Li if li_ops else self.U1T_Idiff_e1l
        self.Idiffmat_mult_r1vecormat = self.Idiff_r1_e1l_Li if li_ops else self.Idiff_r1_e1l

        self.U2mat_mult_r2vec = self.U2r2_e1l_Li if li_ops else self.U2r2_e1l
        self.U2Tmat_mult_L1diffvecormat = self.U2T_L1diff_e1l_Li if li_ops else self.U2T_L1diff_e1l
        self.L1diffvecormat_mult_r2vec = self.L1diff_r2_e1l_Li if li_ops else self.L1diff_r2_e1l

        self.U_gteq3_mat_mult_r_gteq3_vec = self.U3r3_e1l
        self.U3Tmat_mult_L2diffvec = self.U3T_L2diff_e1l
        self.L_gteq2_diffvec_mult_r_gteq3_vec = self.L2diff_r3_e1l
            
    '''
    cost function subcomponents
    
    used in r or U updates
    dependent on architecture parameter
    '''
    
    def U1r1_fhl(self, U1: np.ndarray, r1: np.ndarray) -> np.ndarray:
        """Compute the layer-1 prediction product of U1 and r1 (fhl)."""
        return np.tensordot(U1, r1, axes=([-1], [0]))    

    def U1r1_e1l_Li(self, U1: np.ndarray, r1: np.ndarray) -> np.ndarray:
        """Compute the U1-r1 prediction via batched matmul (e1l, Li reshaped)."""
        return np.matmul(U1, r1[:, :, None]).squeeze()
    
    def U1r1_e1l(self, U1: np.ndarray, r1: np.ndarray) -> np.ndarray:
        """Compute the U1-r1 prediction via einsum (e1l)."""
        # ijk,ik->ij (3d U1, 2d r1); ijklm,ijm->ijkl (5d U1, 3d r1)
        
        return np.einsum(self.U1_einsum_arg, U1, r1)    
    
    def U1T_Idiff_fhl(self, U1T: np.ndarray, Idiff: np.ndarray) -> np.ndarray:
        """Propagate the input-layer error through U1 transpose (fhl)."""
        return np.tensordot(U1T, Idiff, axes=(self.U1T_tdot_dims, self.Idiff_tdot_dims))

    def U1T_Idiff_e1l_Li(self, U1T: np.ndarray, Idiff: np.ndarray) -> np.ndarray:
        """Propagate the input-layer error through U1 transpose (e1l, Li reshaped)."""
        # e.g. U1T (16,32,864) [transpose dims (0,2,1)], Idiff expanded to (16,864,1)
        return np.matmul(U1T, Idiff[:, :, None]).squeeze()
    
    def U1T_Idiff_e1l(self, U1T: np.ndarray, Idiff: np.ndarray) -> np.ndarray:
        """Propagate the input-layer error through U1 transpose (e1l)."""
        # e.g. U1T (32,16,864), Idiff (16,864) -> (16,32)
        # ijk,jk->ji (3d U1, 2d r1); ijklm,jklm->jki (5d U1, 3d r1)
        return np.einsum(self.U1T_einsum_arg, U1T, Idiff)
    
    def U2r2_fhl(self, U2: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Compute the layer-2 prediction product of U2 and r2 (fhl)."""
        return np.dot(U2, r2)

    def U2r2_e1l_Li(self, U2: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Compute the U2-r2 prediction and reshape to the layer-1 shape (e1l, Li reshaped)."""
        return np.dot(U2, r2).reshape(self.r[1].shape)
    
    def U2r2_e1l(self, U2: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Compute the U2-r2 prediction via tensordot (e1l)."""
        return np.tensordot(U2, r2, axes=([-1], [0]))
    
    def U2T_L1diff_fhl(self, U2T: np.ndarray, L1diff: np.ndarray) -> np.ndarray:
        """Propagate the layer-1 error through U2 transpose (fhl)."""
        return np.dot(U2T, L1diff)

    def U2T_L1diff_e1l_Li(self, U2T: np.ndarray, L1diff: np.ndarray) -> np.ndarray:
        """Propagate the flattened layer-1 error through U2 transpose (e1l, Li reshaped)."""
        return np.dot(U2T, L1diff.flatten())
    
    def U2T_L1diff_e1l(self, U2T: np.ndarray, L1diff: np.ndarray) -> np.ndarray:
        """Propagate the layer-1 error through U2 transpose (e1l)."""
        # e.g. U2 (16,32,128) -> U2T (128,16,32), L1diff (16,32): ijk,jk->i
        # or U2 (4,4,32,128) -> U2T (128,4,4,32), L1diff (4,4,32): ijkl,jkl->i
        return np.einsum(self.U2T_einsum_arg, U2T, L1diff)
    
    '''
    these are all dots
    can shortcut later but kept for clarity
    '''
    
    def U3r3_fhl(self, U3: np.ndarray, r3: np.ndarray) -> np.ndarray:
        """Compute a layer >= 3 prediction product (fhl)."""
        return np.dot(U3, r3)

    def U3r3_e1l_Li(self, U3: np.ndarray, r3: np.ndarray) -> np.ndarray:
        """Compute a layer >= 3 prediction product (e1l, Li reshaped)."""
        return np.dot(U3, r3)
    
    def U3r3_e1l(self, U3: np.ndarray, r3: np.ndarray) -> np.ndarray:
        """Compute a layer >= 3 prediction product (e1l)."""
        return np.dot(U3, r3)
    
    def U3T_L2diff_fhl(self, U3T: np.ndarray, L2diff: np.ndarray) -> np.ndarray:
        """Propagate a layer >= 2 error through the weight transpose (fhl)."""
        return np.dot(U3T, L2diff)

    def U3T_L2diff_e1l_Li(self, U3T: np.ndarray, L2diff: np.ndarray) -> np.ndarray:
        """Propagate a layer >= 2 error through the weight transpose (e1l, Li reshaped)."""
        return np.dot(U3T, L2diff)
    
    def U3T_L2diff_e1l(self, U3T: np.ndarray, L2diff: np.ndarray) -> np.ndarray:
        """Propagate a layer >= 2 error through the weight transpose (e1l)."""
        return np.dot(U3T, L2diff)
    
    '''
    cost function subcomponents
    
    used in U updates only
    dependent on architecture parameter
    '''
    
    def Idiff_r1_fhl(self, Idiff: np.ndarray, r1: np.ndarray) -> np.ndarray:
        """Form the U1 gradient outer product of Idiff and r1 (fhl)."""
        # See Idiff_r1_e1l for shape examples.
        return np.einsum(self.Idiff_einsum_arg, Idiff, r1)

    def Idiff_r1_e1l_Li(self, Idiff: np.ndarray, r1: np.ndarray) -> np.ndarray:
        """Form the U1 gradient outer product of Idiff and r1 (e1l, Li reshaped)."""
        return np.matmul(Idiff[:, :, None], r1[:, None, :])
    
    def Idiff_r1_e1l(self, Idiff: np.ndarray, r1: np.ndarray) -> np.ndarray:
        """Form the U1 gradient outer product of Idiff and r1 (e1l)."""
        # e.g. Idiff (16,864), r1 (16,32): ij,ik->ijk (3d U1, 2d r1)
        # 4d tiled input: Idiff (4,4,24,36), r1 (4,4,32), U1 (4,4,24,36,32): ijkl,ijm->ijklm
        return np.einsum(self.Idiff_einsum_arg, Idiff, r1)
    
    def L1diff_r2_fhl(self, L1diff: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Form the U2 gradient outer product of L1diff and r2 (fhl)."""
        return np.outer(L1diff, r2)

    def L1diff_r2_e1l_Li(self, L1diff: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Form the U2 gradient outer product from the flattened layer-1 error (e1l, Li reshaped)."""
        return np.outer(L1diff.flatten(), r2)
    
    def L1diff_r2_e1l(self, L1diff: np.ndarray, r2: np.ndarray) -> np.ndarray:
        """Form the U2 gradient outer product of L1diff and r2 (e1l)."""
        # 16,32     128     ij,k->ijk
        # 4,4,32    128     ijk,l->ijkl
        return np.einsum(self.L1diff_einsum_arg, L1diff, r2)
    
    def L2diff_r3_fhl(self, L2diff: np.ndarray, r3: np.ndarray) -> np.ndarray:
        """Form a layer >= 3 weight-gradient outer product (fhl)."""
        return np.outer(L2diff, r3)

    def L2diff_r3_e1l_Li(self, L2diff: np.ndarray, r3: np.ndarray) -> np.ndarray:
        """Form a layer >= 3 weight-gradient outer product (e1l, Li reshaped)."""
        return np.outer(L2diff, r3)
    
    def L2diff_r3_e1l(self, L2diff: np.ndarray, r3: np.ndarray) -> np.ndarray:
        """Form a layer >= 3 weight-gradient outer product (e1l)."""
        return np.outer(L2diff, r3)
    
    
    '''
    actual cost functions
    '''    
        
    def rep_cost_n_1(self) -> float:
        """Compute the total representation cost for a 1-layer model.

        Sums the weighted input prediction error with the r and U priors.
        """
        r1 = self.r[1]
        U1 = self.U[1]
        bu_tdot_dims = self.Idiff_tdot_dims

        # Bottom-up component of the representation error
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        bu_vec = self.r[0] - fU1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[0]) * bu_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Return total
        return bu_total + prior_r + prior_U
    
    def rep_cost_n_2(self) -> float:
        """Compute the total representation cost for a 2-layer model."""
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        bu_tdot_dims = self.Idiff_tdot_dims
        td_tdot_dims = self.L1diff_tdot_dims

        # Layer 1
        # Bottom-up component of the representation error
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        bu_vec = self.r[0] - fU1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[0]) * bu_square
        
        # Top-down component of the representation error
        fU2r2 = self.f(self.U2mat_mult_r2vec(U2, r2))
        td_vec = r1 - fU2r2
        td_square = np.tensordot(td_vec, td_vec, axes=(td_tdot_dims, td_tdot_dims))
        td_total = (1 / self.ssq[1]) * td_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Layer 2
        # Bottom-up
        bu_total2 = td_total
        
        # Priors on that layer
        prior_r2 = self.g(np.squeeze(r2), self.alph[2])[0]
        prior_U2 = self.h(U2, self.lam[2])[0]
        
        # Return total
        return bu_total + td_total + prior_r + prior_U + bu_total2 + prior_r2 + prior_U2
    
    def rep_cost_n_gt_eq_3(self) -> float:
        """Compute the total representation cost for models with >= 3 layers."""
        
        n = self.num_layers
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        bu_tdot_dims = self.Idiff_tdot_dims
        td_tdot_dims = self.L1diff_tdot_dims

        # Layer 1
        # Bottom-up component of the representation error
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        bu_vec = self.r[0] - fU1r1
        bu_square = np.tensordot(bu_vec, bu_vec, axes=(bu_tdot_dims, bu_tdot_dims))
        bu_total = (1 / self.ssq[0]) * bu_square
        
        # Top-down component of the representation error
        fU2r2 = self.f(self.U2mat_mult_r2vec(U2, r2))
        td_vec = r1 - fU2r2
        td_square = np.tensordot(td_vec, td_vec, axes=(td_tdot_dims, td_tdot_dims))
        td_total = (1 / self.ssq[1]) * td_square
        
        # Priors on that layer
        prior_r = self.g(np.squeeze(r1), self.alph[1])[0]
        prior_U = self.h(U1, self.lam[1])[0]
        
        # Layer 2
        # Bottom-up
        bu_total2 = td_total
        
        # Top-down component
        r3 = self.r[3]
        U3 = self.U[3]
        fU3r3 = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(U3, r3))
        td_vec2 = r2 - fU3r3
        td_square2 = np.dot(td_vec2, td_vec2)
        td_total2 = (1 / self.ssq[2]) * td_square2
        
        # Priors on that layer
        prior_r2 = self.g(np.squeeze(r2), self.alph[2])[0]
        prior_U2 = self.h(U2, self.lam[2])[0]
        
        # Layer 3 - n-1 (skips if 3-layer model)
        bu_totali_all_i = 0
        td_totali_all_i = 0
        prior_ri_all_i = 0
        prior_Ui_all_i = 0
        
        for i in range(3,n):
            ri = self.r[i]
            Ui = self.U[i]
            
            # Layer i bottom-up component
            if i == 3:
                bu_totali = td_total2
            else:
                fUiri = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(Ui, ri))
                bu_veci = self.r[i - 1] - fUiri
                bu_squarei = np.dot(bu_veci, bu_veci)
                bu_totali = (1 / self.ssq[i - 1]) * bu_squarei

            # Priors on that layer
            prior_ri = self.g(np.squeeze(ri), self.alph[i])[0]
            prior_Ui = self.h(Ui, self.lam[i])[0]
            
            # Top-down component
            fUi1ri1 = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(self.U[i + 1], self.r[i + 1]))
            td_veci = ri - fUi1ri1
            td_squarei = np.dot(td_veci, td_veci)
            td_totali = (1 / self.ssq[i]) * td_squarei
            
            bu_totali_all_i += bu_totali
            td_totali_all_i += td_totali
            prior_ri_all_i += prior_ri
            prior_Ui_all_i += prior_Ui
            
        # Layer n
        if n == 3:
            bu_totaln = td_total2
        else:
            fUnrn = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(self.U[n], self.r[n]))
            bun_vec = self.r[n - 1] - fUnrn
            bun_square = np.dot(bun_vec, bun_vec)
            bu_totaln = (1 / self.ssq[n - 1]) * bun_square

        prior_rn = self.g(np.squeeze(self.r[n]), self.alph[n])[0]
        prior_Un = self.h(self.U[n], self.lam[n])[0]
            
        # Return total
        return bu_total + td_total + prior_r + prior_U + bu_total2 + td_total2 + prior_r2 + prior_U2 + \
                bu_totali_all_i + td_totali_all_i + prior_ri_all_i + prior_Ui_all_i + bu_totaln + prior_rn + prior_Un
    
    def classif_cost_c1(self, label: np.ndarray) -> float:
        """Compute the C1 cross-entropy cost: -label.dot(log(softmax(r_n)))."""
        return -label.dot(np.log(self.softmax_func(vector=self.r[self.num_layers])))
    
    def classif_cost_c2(self, label: np.ndarray) -> float:
        """Compute the C2 cross-entropy cost: -label.dot(log(softmax(Uo.dot(r_n))))."""
        return -label.dot(np.log(self.softmax_func(vector=self.U['o'].dot(self.r[self.num_layers]))))
    
    def classif_cost_None(self, label: np.ndarray) -> int:
        """Return zero classification cost (unsupervised)."""
        return 0

    def rW_instdeltas_updates(self, label: np.ndarray, weight_updates: Iterable[Callable[[np.ndarray], None]]) -> None:
        """Apply simultaneous r and U deltas computed from the same pre-update state.

        Computes the r deltas first, runs the in-place weight updates, then
        re-applies both delta sets to the saved pre-update state so r and U
        step "instantaneously" from identical conditions.
        """
        prev_r = {i: np.array(self.r[i], copy=True) for i in range(1, self.num_layers + 1)}
        prev_U = {k: np.array(v, copy=True) for k, v in self.U.items()}

        r_delta = self.r_instdeltas(label)

        for weight_update in weight_updates:
            weight_update(label)
        U_delta = {k: self.U[k] - prev_U[k] for k in prev_U}

        for i in prev_r:
            self.r[i] = prev_r[i] + r_delta[i]
        for k in prev_U:
            self.U[k] = prev_U[k] + U_delta[k]

    def r_instdeltas_update(self, label: np.ndarray) -> None:
        """Apply instantaneous r deltas to every layer in place."""
        r_delta = self.r_instdeltas(label)
        for i in range(1, self.num_layers + 1):
            self.r[i] += r_delta[i]

    def r_instdeltas(self, label: np.ndarray) -> dict[int, np.ndarray]:
        """Compute per-layer instantaneous r deltas without mutating model state.

        Returns:
            Mapping from layer index (1..num_layers) to that layer's r delta.
        """
        n = self.num_layers
        r_prior_cost_denominator = self.r_prior_cost_denominator

        if n == 1:
            r1 = self.r[1]
            U1 = self.U[1]
            kr1 = self.kr[1]
            U1T = np.transpose(U1, self.U1T_dims)
            fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
            Idiff = self.r[0] - fU1r1
            return {1: (kr1 / self.ssq[0]) * self.U1Tmat_mult_Idiffmat(U1T, self.F_mult(fU1r1, Idiff))
                    + (kr1 / self.ssq[1]) * self.rn_topdown_upd_dict[self.classif_method](label)
                    - (kr1 / r_prior_cost_denominator) * self.g(r1, self.alph[1])[1]}

        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        kr1 = self.kr[1]
        kr2 = self.kr[2]
        ssq1 = self.ssq[1]

        U1T = np.transpose(U1, self.U1T_dims)
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        fU2r2 = self.f(self.U2mat_mult_r2vec(U2, r2))
        Idiff = self.r[0] - fU1r1
        L1diff = r1 - fU2r2

        # Layer 1
        r_delta = {1: (kr1 / self.ssq[0]) * self.U1Tmat_mult_Idiffmat(U1T, self.F_mult(fU1r1, Idiff))
                   + (kr1 / ssq1) * (fU2r2 - r1)
                   - (kr1 / r_prior_cost_denominator) * self.g(r1, self.alph[1])[1]}

        # Layer 2 (top layer when n == 2)
        U2T = np.transpose(U2, self.U2T_dims)
        if n == 2:
            r_delta[2] = (kr2 / ssq1) * self.U2Tmat_mult_L1diffvecormat(U2T, self.F_mult(fU2r2, L1diff)) \
                         + (kr2 / self.ssq[2]) * self.rn_topdown_upd_dict[self.classif_method](label) \
                         - (kr2 / r_prior_cost_denominator) * self.g(r2, self.alph[2])[1]
            return r_delta

        r3 = self.r[3]
        U3 = self.U[3]
        fU3r3 = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(U3, r3))
        r_delta[2] = (kr2 / ssq1) * self.U2Tmat_mult_L1diffvecormat(U2T, self.F_mult(fU2r2, L1diff)) \
                     + (kr2 / self.ssq[2]) * (fU3r3 - r2) \
                     - (kr2 / r_prior_cost_denominator) * self.g(r2, self.alph[2])[1]

        # Middle layers 3..n-1
        for i in range(3, n):
            ri = self.r[i]
            ri1 = self.r[i + 1]
            Ui = self.U[i]
            Ui1 = self.U[i + 1]
            kri = self.kr[i]

            UiT = Ui.T
            fUiri = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(Ui, ri))
            fUi1ri1 = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(Ui1, ri1))
            Limin1diff = self.r[i - 1] - fUiri

            r_delta[i] = (kri / self.ssq[i - 1]) * self.U3Tmat_mult_L2diffvec(UiT, self.F_mult(fUiri, Limin1diff)) \
                         + (kri / self.ssq[i]) * (fUi1ri1 - ri) \
                         - (kri / r_prior_cost_denominator) * self.g(ri, self.alph[i])[1]

        # Top layer n: bottom-up + classification + prior terms
        rn = self.r[n]
        Un = self.U[n]
        krn = self.kr[n]
        UnT = Un.T
        fUnrn = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(Un, rn))
        Lnmin1diff = self.r[n - 1] - fUnrn
        r_delta[n] = (krn / self.ssq[n - 1]) * self.U3Tmat_mult_L2diffvec(UnT, self.F_mult(fUnrn, Lnmin1diff)) \
                     + (krn / self.ssq[n]) * self.rn_topdown_upd_dict[self.classif_method](label) \
                     - (krn / r_prior_cost_denominator) * self.g(rn, self.alph[n])[1]

        return r_delta
    
    def rn_topdown_upd_c1(self, label: np.ndarray) -> np.ndarray:
        """Return the C1 classification error term for the top-layer r update."""
        return label - self.softmax_func(vector=self.r[self.num_layers])

    def rn_topdown_upd_c2(self, label: np.ndarray) -> np.ndarray:
        """Return the C2 classification error term, mapped back through Uo.T."""
        class_error = label - self.softmax_func(vector=self.U['o'].dot(self.r[self.num_layers]))
        return self.U['o'].T.dot(class_error)
    
    def rn_topdown_upd_None(self, label: np.ndarray) -> int:
        """Return zero (no classification term)."""
        return 0

    def r_updates_n_1(self, label: np.ndarray) -> None:
        """Update r[1] in place for a 1-layer model."""
        
        r1 = self.r[1]
        U1 = self.U[1]
        kr1 = self.kr[1]
        
        # Layer 1
        U1T = np.transpose(U1, self.U1T_dims)
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        Idiff = self.r[0] - fU1r1
        
        self.r[1] += (kr1 / self.ssq[0]) * self.U1Tmat_mult_Idiffmat(U1T, self.F_mult(fU1r1, Idiff)) \
            + (kr1 / self.ssq[1]) * self.rn_topdown_upd_dict[self.classif_method](label) \
                    - (kr1 / self.r_prior_cost_denominator) * self.g(r1, self.alph[1])[1]
        
    
    def r_updates_n_2(self, label: np.ndarray) -> None:
        """Update r[1] and r[2] in place for a 2-layer model."""
        
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        kr1 = self.kr[1]
        kr2 = self.kr[2]
        ssq1 = self.ssq[1]
        r_prior_cost_denominator = self.r_prior_cost_denominator
        
        # Layer 1
        U1T = np.transpose(U1, self.U1T_dims)
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        fU2r2 = self.f(self.U2mat_mult_r2vec(U2, r2))
        Idiff = self.r[0] - fU1r1
        
        self.r[1] += (kr1 / self.ssq[0]) * self.U1Tmat_mult_Idiffmat(U1T, self.F_mult(fU1r1, Idiff)) \
                    + (kr1 / ssq1) * (fU2r2 - r1) \
                    - (kr1 / r_prior_cost_denominator) * self.g(r1, self.alph[1])[1]
                    
        # Layer 2
        U2T = np.transpose(U2, self.U2T_dims) 
        L1diff = r1 - fU2r2
        
        self.r[2] += (kr2 / ssq1) * self.U2Tmat_mult_L1diffvecormat(U2T, L1diff) \
            + (kr2 / self.ssq[2]) * self.rn_topdown_upd_dict[self.classif_method](label) \
                    - (kr2 / r_prior_cost_denominator) * self.g(r2, self.alph[2])[1]
    
    def r_updates_n_gt_eq_3(self, label: np.ndarray) -> None:
        """Update all layer representations in place for models with >= 3 layers."""
        
        n = self.num_layers
        r1 = self.r[1]
        r2 = self.r[2]
        U1 = self.U[1]
        U2 = self.U[2]
        kr1 = self.kr[1]
        kr2 = self.kr[2]
        ssq1 = self.ssq[1]
        r_prior_cost_denominator = self.r_prior_cost_denominator
        
        # Layer 1
        U1T = np.transpose(U1, self.U1T_dims)
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        fU2r2 = self.f(self.U2mat_mult_r2vec(U2, r2))
        Idiff = self.r[0] - fU1r1
        
        self.r[1] += (kr1 / self.ssq[0]) * self.U1Tmat_mult_Idiffmat(U1T, self.F_mult(fU1r1, Idiff)) \
                    + (kr1 / ssq1) * (fU2r2 - r1) \
                    - (kr1 / r_prior_cost_denominator) * self.g(r1, self.alph[1])[1]
                    
        # Layer 2 - n-1
        for i in range(2, n):
            
            # Layer i == 2
            if i == 2:
                r3 = self.r[3]
                U3 = self.U[3]
                
                U2T = np.transpose(U2, self.U2T_dims) 
                fU3r3 = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(U3, r3))
                L1diff = r1 - fU2r2
                
                self.r[2] += (kr2 / ssq1) * self.U2Tmat_mult_L1diffvecormat(U2T, self.F_mult(fU2r2, L1diff)) \
                            + (kr2 / self.ssq[2]) * (fU3r3 - r2) \
                            - (kr2 / r_prior_cost_denominator) * self.g(r2, self.alph[2])[1]
            
            # Layer i > 2
            else:
                ri = self.r[i]
                ri1 = self.r[i + 1]
                Ui = self.U[i]
                Ui1 = self.U[i + 1]
                kri = self.kr[i]
                
                UiT = Ui.T
                fUiri = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(Ui, ri))
                fUi1ri1 = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(Ui1, ri1))
                Limin1diff = self.r[i - 1] - fUiri
                
                self.r[i] += (kri / self.ssq[i - 1]) * self.U3Tmat_mult_L2diffvec(UiT, self.F_mult(fUiri, Limin1diff)) \
                            + (kri / self.ssq[i]) * (fUi1ri1 - ri) \
                            - (kri / r_prior_cost_denominator) * self.g(ri, self.alph[i])[1]
                    
        # Layer n
        rn = self.r[n]
        Un = self.U[n]
        krn = self.kr[n]
        
        UnT = Un.T
        fUnrn = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(Un, rn))
        Lnmin1diff = self.r[n - 1] - fUnrn
        
        self.r[n] += (krn / self.ssq[n - 1]) * self.U3Tmat_mult_L2diffvec(UnT, self.F_mult(fUnrn, Lnmin1diff)) \
            + (krn / self.ssq[n]) * self.rn_topdown_upd_dict[self.classif_method](label) \
                    - (krn / r_prior_cost_denominator) * self.g(rn, self.alph[n])[1]
    
    def U_updates_n_1(self, label: np.ndarray) -> None:
        """Update U[1] in place for a 1-layer model."""
        
        r1 = self.r[1]
        U1 = self.U[1]
        kU1 = self.kU[1]
        
        # Layer 1
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        Idiff = self.r[0] - fU1r1
        
        self.U[1] += (kU1 / self.ssq[0]) * self.Idiffmat_mult_r1vecormat(self.F_mult(fU1r1, Idiff), r1) \
                    - (kU1 / self.U_prior_cost_denominator) * self.h(U1, self.lam[1])[1]
    
    def U_updates_n_gt_eq_2(self, label: np.ndarray) -> None:
        """Update all generative weights U[1..n] in place for models with >= 2 layers."""
        
        n = self.num_layers
        r1 = self.r[1]
        U1 = self.U[1]
        kU1 = self.kU[1]
        U_prior_cost_denominator = self.U_prior_cost_denominator
        
        # Layer 1
        fU1r1 = self.f(self.U1mat_mult_r1vecormat(U1, r1))
        Idiff = self.r[0] - fU1r1
        
        self.U[1] += (kU1 / self.ssq[0]) * self.Idiffmat_mult_r1vecormat(self.F_mult(fU1r1, Idiff), r1) \
                    - (kU1 / U_prior_cost_denominator) * self.h(U1, self.lam[1])[1]
                    
        # Layer 2 - n
        for i in range(2,n+1):

            # Layer i == 2
            if i == 2:
                r2 = self.r[2]
                U2 = self.U[2]
                kU2 = self.kU[2]
                
                fU2r2 = self.f(self.U2mat_mult_r2vec(self.U[2], self.r[2]))
                L1diff = r1 - fU2r2
                
                self.U[2] += (kU2 / self.ssq[1]) * self.L1diffvecormat_mult_r2vec(self.F_mult(fU2r2, L1diff), r2) \
                            - (kU2 / U_prior_cost_denominator) * self.h(U2, self.lam[2])[1]
            
            # i > 2               
            else:
                ri = self.r[i]
                Ui = self.U[i]
                kUi = self.kU[i]
                
                fUiri = self.f(self.U_gteq3_mat_mult_r_gteq3_vec(Ui, ri))
                Limin1diff = self.r[i - 1] - fUiri
                
                self.U[i] += (kUi / self.ssq[i - 1]) * self.L_gteq2_diffvec_mult_r_gteq3_vec(self.F_mult(fUiri, Limin1diff), ri) \
                            - (kUi / U_prior_cost_denominator) * self.h(Ui, self.lam[i])[1]
    
    def Uo_update(self, label: np.ndarray) -> None:
        """Update the classification output weights U['o'] in place (C2)."""
        # No "Li" denominator option here, because she never ran a C2 model.
        o = 'o'
        n = self.num_layers
        rn = self.r[n]
        self.U[o] += (self.kU[o] / self.ssq[n]) * np.outer((label - self.softmax_func(vector=self.U[o].dot(rn))), rn)

    def classif_guess_c1(self, label: np.ndarray) -> int:
        """Return 1 if the C1 softmax argmax matches the label, else 0."""
        probs = self.softmax_func(vector=self.r[self.num_layers])
        # Treats ties as incorrect (original Li model did as well.)
        if np.sum(probs == probs.max()) != 1:
            return 0
        guess = int(np.argmax(probs))
        return int(guess == int(np.argmax(label)))
    
    def classif_guess_c2(self, label: np.ndarray) -> int:
        """Return 1 if the C2 softmax argmax matches the label, else 0."""
        probs = self.softmax_func(vector=self.U['o'].dot(self.r[self.num_layers]))
        # Treats ties as incorrect (original Li model did as well.)
        if np.sum(probs == probs.max()) != 1:
            return 0
        guess = int(np.argmax(probs))
        return int(guess == int(np.argmax(label)))
        
    def classif_guess_None(self, label: np.ndarray) -> int:
        """Return 0 (no classification guess when unsupervised)."""
        return 0

class RecurrentCostFunction():
    """Cost and gradient-update machinery for a RecurrentPCC model.

    Copies the model's ``__dict__`` at construction. Per-timestep 'bar'
    states are predicted (pre-correction) and 'hat' states are corrected;
    'c' denotes the classification-driven context state at the top layer.
    Recurrent weights V propagate hat states across timesteps.
    """

    def __init__(self, rPCC: Any) -> None:
        """Copy model attributes from a RecurrentPCC instance.

        Args:
            rPCC: RecurrentPCC model instance whose ``__dict__`` is copied.
        """
        # Copy attributes from the model
        self.__dict__.update(rPCC.__dict__)

    def recurrent_classif_error_to_r(self, label: np.ndarray, r_state: np.ndarray, Uo: np.ndarray | None = None) -> np.ndarray | int:
        """Return the classification error term for a recurrent r state.

        C1 applies softmax to r_state directly; C2 applies softmax to
        Uo.dot(r_state) and maps the error back through Uo.T. Returns 0 when
        classif_method is None.
        """
        if self.classif_method == 'c1':
            return self.softmax_dict[self.softmax_type](vector=r_state, k=self.softmax_k) - label
        if self.classif_method == 'c2':
            probs = self.softmax_dict[self.softmax_type](vector=np.matmul(Uo, r_state), k=self.softmax_k)
            return np.matmul(Uo.T, probs - label)
        if self.classif_method is None:
            return 0
        raise ValueError(f'Unsupported classif_method: {self.classif_method}')

    def recurrent_layer_driver(self, layer_index: int, ts: int) -> np.ndarray:
        """Return the corrected (hat) state driving a layer's prediction at ts.

        The top layer is driven by the context ('c') state.
        """
        if layer_index == self.num_layers:
            return self.rhat['c'][:, ts]
        return self.rhat[layer_index][:, ts]

    def recurrent_layer_bar_driver(self, layer_index: int, ts: int) -> np.ndarray:
        """Return the predicted (bar) state for a layer at ts."""
        return self.rbar[layer_index][:, ts]

    def r_updates_n_gteq_1(self, ts: int, label: np.ndarray) -> None:
        """Update all bar/hat representations and the context state at timestep ts."""
        ts_slice_r = (slice(None), ts)
        tsmin1 = 0 if ts - 1 < 0 else ts - 1
        tsmin1_slice_r = (slice(None), tsmin1)
        tsmin1_slice_W = (slice(None), slice(None), tsmin1)

        def previous_recurrent_state(layer_index: int | str) -> np.ndarray:
            """Return the previous-timestep hat state (ts=0 reads the seed slot)."""
            if ts == 0:
                if layer_index == 'c':
                    return self.rhat['c'][:, 0]
                return self.rhat[layer_index][:, 0]
            if layer_index == 'c':
                return self.rhat['c'][tsmin1_slice_r]
            return self.rhat[layer_index][tsmin1_slice_r]

        # Propagate bar states from previous hat states through V
        for i in range(1, self.num_layers + 1):
            self.rbar[i][ts_slice_r] = np.matmul(
                self.Vhat[i][tsmin1_slice_W],
                previous_recurrent_state(i),
            )
        self.rbar['c'][ts_slice_r] = np.matmul(
            self.Vhat[self.num_layers][tsmin1_slice_W],
            previous_recurrent_state('c'),
        )

        # Correct each layer's hat state with bottom-up (and top-down) errors
        for i in range(1, self.num_layers + 1):
            Ubarits = self.Uhat[i][tsmin1_slice_W]
            rbartits = self.rbar[i][ts_slice_r]
            fUr = self.f(np.matmul(Ubarits, rbartits))
            if i == 1:
                bottom_error = self.r[0] - fUr
            else:
                bottom_error = self.rbar[i - 1][ts_slice_r] - fUr

            self.rhat[i][ts_slice_r] = rbartits + (self.kr[i] / self.ssqr[i - 1]) * np.matmul(Ubarits.T, self.F_mult(fUr, bottom_error))

            if i < self.num_layers:
                Ubar_next = self.Uhat[i + 1][tsmin1_slice_W]
                top_prediction = self.f(np.matmul(Ubar_next, self.rbar[i + 1][ts_slice_r]))
                self.rhat[i][ts_slice_r] += (self.kr[i] / self.ssqr[i]) * (top_prediction - rbartits)

        # Context state: apply the classification error to the top-layer hat
        Ubarots = self.Uhat['o'][tsmin1_slice_W] if self.classif_method == 'c2' else None
        top_bar = self.rbar[self.num_layers][ts_slice_r]
        classif_error_to_r = self.recurrent_classif_error_to_r(label, top_bar, Uo=Ubarots)
        self.rhat['c'][ts_slice_r] = self.rhat[self.num_layers][ts_slice_r] - (
            (1 / self.rc_topdown_cost_denominator) * (self.kr[self.num_layers] / self.ssqr[self.num_layers]) * classif_error_to_r
        )
    
    def r_updates_n_2(self, ts: int, label: np.ndarray) -> None:
        """Update 2-layer bar/hat representations and the context state at timestep ts."""
        # time slices
        ts_slice_r = (slice(None), ts) # [:, ts]
        if ts - 1 < 0:
            tsmin1 = 0
        else:
            tsmin1 = ts - 1
        tsmin1_slice_r = (slice(None), tsmin1) # [:, ts - 1]
        tsmin1_slice_W = (slice(None), slice(None), tsmin1) # [:, :, ts - 1]

        if ts == 0:
            prev_rhat1 = self.rhat[1][:, 0]
            prev_rhat2 = self.rhat[2][:, 0]
            prev_rhatc = self.rhat['c'][:, 0]
        else:
            prev_rhat1 = self.rhat[1][tsmin1_slice_r]
            prev_rhat2 = self.rhat[2][tsmin1_slice_r]
            prev_rhatc = self.rhat['c'][tsmin1_slice_r]
        
        # r bars first
        # Sequence of hats and bars to do with the fact that r, U, V is the global update order.
        rbar1ts = self.rbar[1][ts_slice_r] = np.matmul(self.Vhat[1][tsmin1_slice_W], prev_rhat1)
        rbar2ts = self.rbar[2][ts_slice_r] = np.matmul(self.Vhat[2][tsmin1_slice_W], prev_rhat2)
        # r_context bar: uses same V2, propagates r_context from previous ts (C2 fix)
        rbarcts = self.rbar['c'][ts_slice_r] = np.matmul(self.Vhat[2][tsmin1_slice_W], prev_rhatc)
        
        Ubar1ts = self.Uhat[1][tsmin1_slice_W]
        Ubar2ts = self.Uhat[2][tsmin1_slice_W]
        Ubarots = self.Uhat['o'][tsmin1_slice_W] if self.classif_method == 'c2' else None
        
        kr1 = self.kr[1]
        kr2 = self.kr[2]
        
        ssqr1 = self.ssqr[1]
        
        # r hats
        fU1rbar1 = self.f(np.matmul(Ubar1ts, rbar1ts))
        fU2rbar2 = self.f(np.matmul(Ubar2ts, rbar2ts))
        self.rhat[1][ts_slice_r] = rbar1ts + (kr1 / self.ssqr[0]) \
                                * np.matmul(Ubar1ts.T, self.F_mult(fU1rbar1, self.r[0] - fU1rbar1)) \
                                - (kr1 / ssqr1) * (rbar1ts - fU2rbar2)
        
        # Layer 2                     
        self.rhat[2][ts_slice_r] = rbar2ts + (kr2 / ssqr1) \
                                * np.matmul(Ubar2ts.T, self.F_mult(fU2rbar2, rbar1ts - fU2rbar2))
        
        # r_context (r2_hat_x in inext): softmax uses softmax_k (training k; baseline 1), not softmax_k_eval
        classif_error_to_r = self.recurrent_classif_error_to_r(label, rbar2ts, Uo=Ubarots)
        self.rhat['c'][ts_slice_r] = self.rhat[2][ts_slice_r] - (1 / self.rc_topdown_cost_denominator) * (kr2 / self.ssqr[2]) * \
                    classif_error_to_r

    def U_updates_n_2(self, ts: int, label: np.ndarray) -> None:
        """Update 2-layer Ubar/Uhat generative weights at timestep ts."""
        # time slices
        ts_slice_r = (slice(None), ts) # [:, ts]
        ts_slice_W = (slice(None), slice(None), ts) # [:, :, ts]
        if ts - 1 < 0:
            tsmin1 = 0
        else:
            tsmin1 = ts - 1
        tsmin1_slice_W = (slice(None), slice(None), tsmin1) # [:, :, ts - 1]
        
        # U bars first: Ubar(t) = Uhat(t-1)
        Ubar1ts = self.Ubar[1][ts_slice_W] = copy(self.Uhat[1][tsmin1_slice_W])
        Ubar2ts = self.Ubar[2][ts_slice_W] = copy(self.Uhat[2][tsmin1_slice_W])
        c = 'c'
        
        rhat1ts = self.rhat[1][ts_slice_r]
        rhatcts = self.rhat[c][ts_slice_r]
        
        # U hats
        fU1rhat1 = self.f(np.matmul(Ubar1ts, rhat1ts))
        fU2rhatc = self.f(np.matmul(Ubar2ts, rhatcts))
        self.Uhat[1][ts_slice_W] = Ubar1ts + (self.kU[1] / self.ssqr[0]) \
                                    * np.outer(self.F_mult(fU1rhat1, self.r[0] - fU1rhat1), rhat1ts)
                                    
        self.Uhat[2][ts_slice_W] = Ubar2ts + (self.kU[2] / self.ssqr[1]) \
                                    * np.outer(self.F_mult(fU2rhatc, self.rbar[1][ts_slice_r] - fU2rhatc), rhatcts)

    def U_updates_n_gteq_1(self, ts: int, label: np.ndarray) -> None:
        """Update Ubar/Uhat generative weights for all layers at timestep ts."""
        ts_slice_r = (slice(None), ts)
        ts_slice_W = (slice(None), slice(None), ts)
        tsmin1 = 0 if ts - 1 < 0 else ts - 1
        tsmin1_slice_W = (slice(None), slice(None), tsmin1)

        for i in range(1, self.num_layers + 1):
            Ubarits = self.Ubar[i][ts_slice_W] = copy(self.Uhat[i][tsmin1_slice_W])
            driver = self.recurrent_layer_driver(i, ts)
            fUdriver = self.f(np.matmul(Ubarits, driver))
            if i == 1:
                prediction_error = self.r[0] - fUdriver
            else:
                prediction_error = self.rbar[i - 1][ts_slice_r] - fUdriver
            self.Uhat[i][ts_slice_W] = Ubarits + (self.kU[i] / self.ssqr[i - 1]) * np.outer(self.F_mult(fUdriver, prediction_error), driver)

    def Uo_update(self, ts: int, label: np.ndarray) -> None:
        """Update the classification weights Uhat['o'] at timestep ts (C2 only)."""
        if self.classif_method != 'c2':
            return

        ts_slice_r = (slice(None), ts)
        ts_slice_W = (slice(None), slice(None), ts)
        tsmin1 = 0 if ts - 1 < 0 else ts - 1
        tsmin1_slice_W = (slice(None), slice(None), tsmin1)

        Ubarots = self.Ubar['o'][ts_slice_W] = copy(self.Uhat['o'][tsmin1_slice_W])
        rhatcts = self.rhat['c'][ts_slice_r]
        class_error = label - self.softmax_dict[self.softmax_type](vector=np.matmul(Ubarots, rhatcts), k=self.softmax_k)
        self.Uhat['o'][ts_slice_W] = Ubarots + (self.kU['o'] / self.ssqr[self.num_layers]) * np.outer(class_error, rhatcts)
                                    
    def V_updates_n_2(self, ts: int, label: np.ndarray) -> None:
        """Update 2-layer Vbar/Vhat recurrent weights at timestep ts."""
        # time slices
        ts_slice_r = (slice(None), ts) # [:, ts]
        ts_slice_W = (slice(None), slice(None), ts) # [:, :, ts]
        tsmin1_slice_W = (slice(None), slice(None), 0 if ts == 0 else ts - 1)

        # Bars first
        Vbar1ts = self.Vbar[1][ts_slice_W] = copy(self.Vhat[1][tsmin1_slice_W])
        Vbar2ts = self.Vbar[2][ts_slice_W] = copy(self.Vhat[2][tsmin1_slice_W])

        # "Previous" r values for the outer-product term.
        # For ts=0 this is the explicit per-input reset state used to seed
        # rhat trajectories (generalizes inext's zero-start shortcut).
        # We cannot use rhat[:, ts=0] here because r_updates has already
        # overwritten that slot for the current timestep.
        if ts == 0:
            rhat1_prev = self.r[1]
            rhatc_prev = self.r['c']
        else:
            rhat1_prev = self.rhat[1][:, ts - 1]
            rhatc_prev = self.rhat['c'][:, ts - 1]

        # Hats
        self.Vhat[1][ts_slice_W] = Vbar1ts + (self.kV[1] / self.ssqV[1]) \
                                    * np.outer((self.rhat[1][ts_slice_r] - self.rbar[1][ts_slice_r]), rhat1_prev)
        # V2 update uses r_context (r2_hat_x in inext) throughout — C1 fix
        self.Vhat[2][ts_slice_W] = Vbar2ts + (self.kV[2] / self.ssqV[2]) \
                                    * np.outer((self.rhat['c'][ts_slice_r] - self.rbar['c'][ts_slice_r]), rhatc_prev)

    def V_updates_n_gteq_1(self, ts: int, label: np.ndarray) -> None:
        """Update Vbar/Vhat recurrent weights for all layers at timestep ts."""
        ts_slice_r = (slice(None), ts)
        ts_slice_W = (slice(None), slice(None), ts)
        tsmin1_slice_W = (slice(None), slice(None), 0 if ts == 0 else ts - 1)

        for i in range(1, self.num_layers + 1):
            Vbarits = self.Vbar[i][ts_slice_W] = copy(self.Vhat[i][tsmin1_slice_W])
            if i == self.num_layers:
                current_state = self.rhat['c'][ts_slice_r]
                current_bar = self.rbar['c'][ts_slice_r]
                previous_state = self.r['c'] if ts == 0 else self.rhat['c'][:, ts - 1]
            else:
                current_state = self.rhat[i][ts_slice_r]
                current_bar = self.rbar[i][ts_slice_r]
                previous_state = self.r[i] if ts == 0 else self.rhat[i][:, ts - 1]
            self.Vhat[i][ts_slice_W] = Vbarits + (self.kV[i] / self.ssqV[i]) * np.outer(
                current_state - current_bar,
                previous_state,
            )
                                    
    def rep_cost_n_2(self, input: np.ndarray, num_ts: int) -> float:
        """Accumulate the 2-layer representation cost across timesteps."""
        
        bu_total = 0
        td_total = 0      
        bu_total2 = 0
        
        # Uhat and rbar for representation cost
        for ts in range(num_ts):
            # L1
            # Input
            self.r[0] = input[:, ts]
            
            # Bottom-Up
            bu_vec = self.r[0] - self.f(np.matmul(self.Uhat[1][:, :, ts], self.rbar[1][:, ts]))
            bu_square = np.dot(bu_vec, bu_vec)
            bu_total += (1 / self.ssqr[0]) * bu_square
            # Top-Down
            td_vec = self.rbar[1][:, ts] - self.f(np.matmul(self.Uhat[2][:, :, ts], self.rbar[2][:, ts]))
            td_square = np.dot(td_vec, td_vec)
            td_total += (1 / self.ssqr[1]) * td_square

            #L2
            # Bottom-up
            bu_total2 += td_total
        
        # Return total
        return bu_total + td_total + bu_total2

    def rep_cost_n_gteq_1(self, input: np.ndarray, num_ts: int) -> float:
        """Accumulate the n-layer representation cost across timesteps."""
        total = 0

        for ts in range(num_ts):
            self.r[0] = input[:, ts]
            input_error = self.r[0] - self.f(np.matmul(self.Uhat[1][:, :, ts], self.rbar[1][:, ts]))
            total += (1 / self.ssqr[0]) * np.dot(input_error, input_error)

            for i in range(2, self.num_layers + 1):
                interlayer_error = self.rbar[i - 1][:, ts] - self.f(np.matmul(
                    self.Uhat[i][:, :, ts],
                    self.rbar[i][:, ts],
                ))
                error_cost = (1 / self.ssqr[i - 1]) * np.dot(interlayer_error, interlayer_error)
                total += 2 * error_cost

        return total
    
    def classif_cost_c1(self, label: np.ndarray, num_ts: int) -> float:
        """Sum the C1 cross-entropy cost on rbar_n over timesteps."""
        # Training-k on rbar_n: Jc mirrors the objective/state whose gradient drives learning.
        c1tot = 0
        for ts in range(num_ts):
            c1tot += -label.dot(np.log(self.softmax_func(vector=self.rbar[self.num_layers][:, ts])))
        return c1tot

    def classif_cost_c2(self, label: np.ndarray, num_ts: int) -> float:
        """Sum the C2 cross-entropy cost on Uhat['o']-projected rbar_n over timesteps."""
        c2tot = 0
        for ts in range(num_ts):
            logits = np.matmul(self.Uhat['o'][:, :, ts], self.rbar[self.num_layers][:, ts])
            c2tot += -label.dot(np.log(self.softmax_func(vector=logits)))
        return c2tot

    def classif_cost_None(self, label: np.ndarray, num_ts: int) -> int:
        """Return zero classification cost (unsupervised)."""
        return 0
    
    def classif_guess_c1(self, label: np.ndarray, num_ts: int) -> int:
        """Return 1 if the final-timestep C1 guess matches the label, else 0."""
        # Scores the corrected state rhat_n, matching legacy recognition on softmaxd_r2_hat.
        probs = self.softmax_func_eval(vector=self.rhat[self.num_layers][:, num_ts - 1])
        # Legacy ext behavior used argmax tie-breaking; baseline treats ties as incorrect.
        if np.sum(probs == probs.max()) != 1:
            return 0
        guess = int(np.argmax(probs))
        return int(guess == int(np.argmax(label)))

    def classif_guess_c2(self, label: np.ndarray, num_ts: int) -> int:
        """Return 1 if the final-timestep C2 guess matches the label, else 0."""
        logits = np.matmul(self.Uhat['o'][:, :, num_ts - 1], self.rhat[self.num_layers][:, num_ts - 1])
        probs = self.softmax_func_eval(vector=logits)
        if np.sum(probs == probs.max()) != 1:
            return 0
        guess = int(np.argmax(probs))
        return int(guess == int(np.argmax(label)))

    def classif_guess_None(self, label: np.ndarray, num_ts: int) -> int:
        """Return 0 (no classification guess when unsupervised)."""
        return 0