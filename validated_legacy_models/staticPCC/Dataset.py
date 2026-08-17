# -*- coding: utf-8 -*-
"""Image dataset builder for the legacy static PCC models.

Load PNG images, optionally filter them (difference-of-Gaussians or linear
rescale), cut them into RF2 patches with one-hot image-identity labels, and
serve overlapping RF1 sub-patches to the model. This is part of the validated
legacy reference implementation of the Li/Rogers static PCC model.
"""

from __future__ import annotations

import numpy as np
import cv2
import os
import glob


class Dataset:
    """Patch dataset of filtered images for static PCC training and evaluation."""

    def __init__(self, scale: float = 10.0, shuffle: bool = False, data_dir: str | None = None,
                 rf1_x: int = 16, rf1_y: int = 16, rf1_offset_x: int = 5, rf1_offset_y: int = 5,
                 rf1_layout_x: int = 3, rf1_layout_y: int = 1,
                 use_mask: bool = True, gauss_mask_sigma: float = 0.4,
                 image_filter: str | tuple[float, float] | None = "DoG", DoG_ksize: tuple[int, int] = (5,5),
                 DoG_sigma1: float = 1.3, DoG_sigma2: float = 2.6) -> None:
        """Load images, build RF2 patches and labels, and prepare the RF1 Gaussian mask.

        Args:
            image_filter: "DoG", a numeric (min, max) rescale range, or any
                other value to disable filtering.
        """
        self.rf1_size = (rf1_y, rf1_x)
        self.rf1_layout_size = (rf1_layout_y, rf1_layout_x)
        self.rf2_size = (rf1_y+(rf1_layout_y-1)*rf1_offset_y,
                         rf1_x+(rf1_layout_x-1)*rf1_offset_x)
        self.rf1_offset_x = rf1_offset_x
        self.rf1_offset_y = rf1_offset_y

        # Accept "DoG" or a numeric 2-tuple rescale range; anything else disables filtering.
        if image_filter == "DoG":
            self.image_filter = image_filter
        elif (type(image_filter) == tuple):
            if (len(image_filter) == 2) & (all([type(x) in [int, float] for x in image_filter])):
                self.image_filter = image_filter
            else:
                self.image_filter = None
        else:
            self.image_filter = None
        self.DoG_ksize = DoG_ksize
        self.DoG_sigma1 = DoG_sigma1
        self.DoG_sigma2 = DoG_sigma2

        self.load_images(scale, data_dir, image_filter)

        if shuffle:
            indices = np.random.permutation(len(self.rf2_patches))
            self.rf2_patches = self.rf2_patches[indices]
            self.labels = self.labels[indices]

        self.use_mask = use_mask
        self.gauss_mask_sigma = gauss_mask_sigma
        self.mask = self.create_gauss_mask(sigma=gauss_mask_sigma,
                                           width=rf1_x,
                                           height=rf1_y)

    def load_images(self, scale: float, data_dir: str | None,
                    image_filter: str | tuple[float, float] | None) -> None:
        """Load all PNG images from the default or user-specified directory."""
        images = []

        if data_dir == None:
            # Default to the Rao & Ballard paper images bundled next to this script.
            script_dir = os.path.dirname(os.path.realpath(__file__))
            dir_name = os.path.join(script_dir, "data", "raw", "images_rao_128x128")
        else:
            dir_name = data_dir

        file_names = sorted(glob.glob(os.path.join(dir_name, "*.png")))

        for i in file_names:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            images.append(image)

        images = np.array(images)
        self.load_sub(images, scale, image_filter)

    def create_gauss_mask(self, sigma: float = 0.4, width: int = 16, height: int = 16) -> np.ndarray:
        """Create a 2-D Gaussian mask normalized to peak at 1."""
        mu = 0.0
        x, y = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
        d = np.sqrt(x**2+y**2)
        g = np.exp(-( (d-mu)**2 / (2.0*sigma**2) )) / np.sqrt(2.0*np.pi*sigma**2)
        mask = g / np.max(g)
        return mask

    def load_sub(self, images: np.ndarray, scale: float,
                 image_filter: str | tuple[float, float] | None) -> None:
        """Filter images and tile each into non-overlapping RF2 patches with one-hot labels."""
        rf2_x = self.rf2_size[1]
        rf2_y = self.rf2_size[0]

        self.images = images

        filtered_images = []
        for image in images:
            if self.image_filter == None:
                filtered_image = image
            elif self.image_filter == "DoG":
                filtered_image = self.apply_DoG_filter(image, ksize=self.DoG_ksize, sigma1=self.DoG_sigma1, sigma2=self.DoG_sigma2)
            else:
                filtered_image = np.interp(image, (0, 255), image_filter)
            filtered_images.append(filtered_image)

        self.filtered_images = filtered_images

        w = images.shape[2]
        h = images.shape[1]

        size_w = w // rf2_x
        size_h = h // rf2_y

        rf2_patches = np.empty((size_h * size_w * len(images), rf2_y, rf2_x), dtype=np.float32)
        # different patches of the same training image will be assigned the same identity
        labels = np.empty((size_h * size_w * len(images), len(images)), dtype=np.float32)

        for image_index, filtered_image in enumerate(filtered_images):
            for i in range(size_w):
                for j in range(size_h):
                    x = rf2_x * i
                    y = rf2_y * j
                    rf2_patch = filtered_image[y:y+rf2_y, x:x+rf2_x]
                    index = size_w*size_h*image_index + j*size_w + i
                    rf2_patches[index] = rf2_patch
                    labels[index] = np.identity(len(images))[image_index]

        self.rf2_patches = rf2_patches * scale
        self.labels = labels

    def get_rf1_patches_from_rf2_patch(self, rf2_patch: np.ndarray) -> list[np.ndarray]:
        """Slice an RF2 patch into flattened, optionally Gaussian-masked RF1 patches."""
        rf1_x = self.rf1_size[1]
        rf1_y = self.rf1_size[0]
        rf1_offset_x = self.rf1_offset_x
        rf1_offset_y = self.rf1_offset_y
        rf1_layout_x = self.rf1_layout_size[1]
        rf1_layout_y = self.rf1_layout_size[0]

        rf1_patches = []
        for i in range(rf1_layout_x):
            for j in range(rf1_layout_y):
                x = rf1_offset_x * i
                y = rf1_offset_y * j
                # Slice the RF1 patch, then optionally apply the Gaussian mask.
                rf1_patch = rf2_patch[y:y+rf1_y, x:x+rf1_x].reshape([-1])
                if self.use_mask:
                    rf1_patch = rf1_patch * self.mask.reshape([-1])
                rf1_patches.append(rf1_patch)
        return rf1_patches

    def get_rf1_patches(self, rf2_patch_index: int) -> list[np.ndarray]:
        """Return the RF1 patches for the RF2 patch at the given index."""
        rf2_patch = self.rf2_patches[rf2_patch_index]
        return self.get_rf1_patches_from_rf2_patch(rf2_patch)

    def apply_DoG_filter(self, image: np.ndarray, ksize: tuple[int, int] = (5,5),
                         sigma1: float = 1.3, sigma2: float = 2.6) -> np.ndarray:
        """Apply a difference-of-Gaussians (DoG) filter to edge-detect the image."""
        g1 = cv2.GaussianBlur(image, ksize, sigma1)
        g2 = cv2.GaussianBlur(image, ksize, sigma2)
        return g1 - g2

    # Bar-stimulus helpers from the original implementation, retained commented-out and unused.
    # def get_bar_images(self, is_short):
    #     patch = self.get_bar_patch(is_short)
    #     return self.get_images_from_patch(patch, use_mask=True)

    # def get_bar_patch(self, is_short):
    #     """
    #     Get bar patch image for end stopping test.
    #     """
    #     bar_patch = np.ones((16,26), dtype=np.float32)

    #     if is_short:
    #         bar_width = 6
    #     else:
    #         bar_width = 24
    #     bar_height = 2

    #     for x in range(bar_patch.shape[1]):
    #         for y in range(bar_patch.shape[0]):
    #             if x >= 26/2 - bar_width/2 and \
    #             x < 26/2 + bar_width/2 and \
    #             y >= 16/2 - bar_height/2 and \
    #             y < 16/2 + bar_height/2:
    #                 bar_patch[y,x] = -1.0

    #     # Sete scale with stddev of all patch images.
    #     scale = np.std(self.patches)
    #     # Original scaling value for bar
    #     bar_scale = 2.0
    #     return bar_patch * scale * bar_scale
