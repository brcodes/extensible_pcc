"""Preprocess raw stimuli into pickled (X, Y) datasets for pypredcoding models.

Convert raw images/cochleograms under data/raw/ into .pydb pickles containing
inputs X and one-hot labels Y, in the shapes expected by the recurrent (rPCC)
and static (sPCC) predictive-coding models. Invoke from this directory as
``python data.py --cvcv12 | --tom3phon3let_16set | --tom3phon3let_4set |
--raonaturalimages5 | --trace212``.
"""

from __future__ import annotations

import glob
import os
import pickle
import re
import sys
from pathlib import Path
from ast import literal_eval
from typing import Any

import cv2
import numpy as np


SPCC_DATASET_SPECS = {
    'Rogers_2026_raonaturalimages5': {
        'dataset_name': 'Rogers_2026_raonaturalimages5',
        'raw_dir': 'images_rao_128x128',
        'input_scale': 1.0,
        'rf1_x': 16,
        'rf1_y': 16,
        'rf1_offset_x': 8,
        'rf1_offset_y': 8,
        'rf1_layout_x': 15,
        'rf1_layout_y': 15,
        'use_mask': True,
        'gauss_mask_sigma': 1.0,
        'image_filter': 'DoG',
        'dog_ksize': (5, 5),
        'dog_sigma1': 1.3,
        'dog_sigma2': 2.6,
    },
    'Rogers_2026_trace212': {
        'dataset_name': 'Rogers_2026_trace212',
        'raw_dir': 'trace212',
        'input_scale': 1.0,
        'rf1_x': 36,
        'rf1_y': 24,
        'rf1_offset_x': 32,
        'rf1_offset_y': 20,
        'rf1_layout_x': 4,
        'rf1_layout_y': 4,
        'use_mask': False,
        'gauss_mask_sigma': 1.0,
        'image_filter': 'DoG',
        'dog_ksize': (5, 5),
        'dog_sigma1': 1.3,
        'dog_sigma2': 2.6,
    },
}


def preprocess_cvcv12_for_rpcc() -> tuple[np.ndarray, np.ndarray]:
    """Preprocess raw CVCV_12 PNG images into the rPCC (X, Y) format.

    Load PNGs from data/raw/CVCV_12 in sorted filename order, convert each
    BGR image to grayscale normalized to [0, 1], and key it by the word ID
    embedded in the filename (substring after the last '_' before '.png').
    Save (X, Y) to pypredcoding/data/Rogers_2026_cvcv12.pydb.

    Returns:
        Tuple of X with shape (N_words, I_size, num_ts) and Y as an
        (N_words, N_words) identity matrix of one-hot labels.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_dir = os.path.join(script_dir, "data", "raw", "CVCV_12")
    out_path = os.path.join(script_dir, "data", "Rogers_2026_cvcv12.pydb")

    filenames = sorted(glob.glob(os.path.join(in_dir, "*.png")))
    if not filenames:
        raise FileNotFoundError(f"No PNG files found in {in_dir}")

    image_by_word = {}
    for filepath in filenames:
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError(f"Unable to read image: {filepath}")

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray_img /= 255.0

        word_id = re.sub(r".*_(.*)\.png", r"\1", filepath)
        image_by_word[word_id] = gray_img

    words = list(image_by_word.keys())
    x_data = np.stack([image_by_word[word] for word in words], axis=0).astype(np.float64)
    y_data = np.eye(len(words), dtype=np.float64)

    print(f"Words ({len(words)}): {words}")
    print(f"X shape: {x_data.shape}  (N_words, I_size, num_ts)")
    print(f"Y shape: {y_data.shape}  (N_words, N_words)")

    with open(out_path, "wb") as out_file:
        pickle.dump((x_data, y_data), out_file)

    print(f"Saved: {out_path}")
    return x_data, y_data


# all have at least 1 cohort, 1 rhyme, 1 unrelated
TOM3PHON3LET_16SET_WORDS = [
    "DIM",
    "DIG",
    "PIG",
    "PIT",
    "SIT",
    "SIN",
    "PIN",
    "RIM",
    "RIB",
    "BIB",
    "BIG",
    "WIG",
    "WIN",
    "DIP",
    "TIP",
    "TIN",
]

# all have exactly 1 cohort, 1 rhyme, 1 unrelated 
TOM3PHON3LET_4SET_WORDS = ["LOT", "LOP", "TOP", "TOT"]


def preprocess_tom3phon3let_for_rpcc(lexicon_choice: list[str] = TOM3PHON3LET_16SET_WORDS) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess selected TOM 3-phoneme/3-letter cochleogram pickles for rPCC.

    Load the selected word list from data/raw/tom_3phoneme_3letter_cochleograms
    and pad each (94, num_ts) cochleogram with trailing NaN columns to the
    set's maximum num_ts. Save (X, Y) to a set-named .pydb in data/.

    Returns:
        Tuple of X with shape (N_words, I_size, num_ts) and Y as an
        (N_words, N_words) identity matrix of one-hot labels.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    in_dir = os.path.join(script_dir, "data", "raw", "tom_3phoneme_3letter_cochleograms")

    lexicon_words = list(lexicon_choice)
    if lexicon_words == TOM3PHON3LET_16SET_WORDS:
        set_name = "16set"
    elif lexicon_words == TOM3PHON3LET_4SET_WORDS:
        set_name = "4set"
    else:
        set_name = "custom"

    out_path = os.path.join(script_dir, "data", f"tom_3phon3let_{set_name}_nanpad.pydb")

    word_arrays = {}
    for word in lexicon_words:
        filepath = os.path.join(in_dir, f"{word}_TOM.pkl")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tom 3phon3let word pickle not found: {filepath}")

        with open(filepath, "rb") as infile:
            word_array = np.asarray(pickle.load(infile), dtype=np.float64)

        if word_array.ndim != 2:
            raise ValueError(f"{filepath} must contain a 2D cochleogram, got shape {word_array.shape}.")
        if word_array.shape[0] != 94:
            raise ValueError(f"{filepath} must have 94 frequency rows, got shape {word_array.shape}.")

        word_arrays[word] = word_array

    max_num_ts = max(word_array.shape[1] for word_array in word_arrays.values())
    padded_word_arrays = []
    for word in lexicon_words:
        word_array = word_arrays[word]
        num_pad_cols = max_num_ts - word_array.shape[1]
        padded_word_array = np.pad(
            word_array,
            ((0, 0), (0, num_pad_cols)),
            mode="constant",
            constant_values=np.nan,
        )
        padded_word_arrays.append(padded_word_array)

    x_data = np.stack(padded_word_arrays, axis=0).astype(np.float64)
    y_data = np.eye(len(lexicon_words), dtype=np.float64)

    print(f"Words ({len(lexicon_words)}): {lexicon_words}")
    print(f"X shape: {x_data.shape}  (N_words, I_size, num_ts)")
    print(f"Y shape: {y_data.shape}  (N_words, N_words)")

    with open(out_path, "wb") as out_file:
        pickle.dump((x_data, y_data), out_file)

    print(f"Saved: {out_path}")
    return x_data, y_data


def _create_gauss_mask(sigma: float = 0.4, width: int = 16, height: int = 16) -> np.ndarray:
    """Build a 2D Gaussian mask over [-1, 1] x [-1, 1], normalized to peak at 1.0."""
    mu = 0.0
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    d = np.sqrt(x**2 + y**2)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2))) / np.sqrt(2.0 * np.pi * sigma**2)
    return g / np.max(g)


def _apply_image_filter(image: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    """Apply the spec's image filter: None (passthrough), 'DoG', or a (lo, hi) rescale tuple.

    Raises:
        ValueError: If the filter spec is not one of the supported forms.
    """
    image_filter = spec['image_filter']
    if image_filter is None:
        return image
    if image_filter == 'DoG':
        return _apply_dog_filter(
            image,
            ksize=tuple(spec['dog_ksize']),
            sigma1=float(spec['dog_sigma1']),
            sigma2=float(spec['dog_sigma2']),
        )
    if isinstance(image_filter, tuple) and len(image_filter) == 2:
        return np.interp(image, (0, 255), image_filter)
    raise ValueError(f"Unsupported static image_filter: {image_filter!r}")


def _apply_dog_filter(image: np.ndarray, ksize: tuple[int, int] = (5, 5), sigma1: float = 1.3, sigma2: float = 2.6) -> np.ndarray:
    """Apply a Difference-of-Gaussians band-pass filter to an image."""
    g1 = cv2.GaussianBlur(image, ksize, sigma1)
    g2 = cv2.GaussianBlur(image, ksize, sigma2)
    return g1 - g2


def _load_static_png_images(raw_dir: str) -> np.ndarray:
    """Load all PNGs in raw_dir (sorted by filename) as grayscale float32 images."""
    file_names = sorted(glob.glob(os.path.join(raw_dir, '*.png')))
    if not file_names:
        raise FileNotFoundError(f'No PNG files found in {raw_dir}')

    images = []
    for file_name in file_names:
        image = cv2.imread(file_name)
        if image is None:
            raise ValueError(f'Unable to read image: {file_name}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        images.append(image)
    return np.asarray(images, dtype=np.float32)


def _extract_rf2_patches(filtered_images: np.ndarray, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tile each filtered image into non-overlapping RF2-sized patches.

    The RF2 extent is the full span of the RF1 tiling defined by the spec.

    Returns:
        Tuple of RF2 patches scaled by input_scale, shape (N_patches, rf2_y, rf2_x),
        and one-hot labels identifying each patch's source image.
    """
    rf2_x = spec['rf1_x'] + (spec['rf1_layout_x'] - 1) * spec['rf1_offset_x']
    rf2_y = spec['rf1_y'] + (spec['rf1_layout_y'] - 1) * spec['rf1_offset_y']

    width = filtered_images.shape[2]
    height = filtered_images.shape[1]
    size_w = width // rf2_x
    size_h = height // rf2_y

    rf2_patches = np.empty((size_h * size_w * len(filtered_images), rf2_y, rf2_x), dtype=np.float32)
    labels = np.empty((size_h * size_w * len(filtered_images), len(filtered_images)), dtype=np.float32)

    for image_index, filtered_image in enumerate(filtered_images):
        for patch_x in range(size_w):
            for patch_y in range(size_h):
                x_start = rf2_x * patch_x
                y_start = rf2_y * patch_y
                rf2_patch = filtered_image[y_start:y_start + rf2_y, x_start:x_start + rf2_x]
                index = size_w * size_h * image_index + patch_y * size_w + patch_x
                rf2_patches[index] = rf2_patch
                labels[index] = np.eye(len(filtered_images), dtype=np.float32)[image_index]

    return rf2_patches * float(spec['input_scale']), labels


def _rf1_patches_from_rf2_patch(rf2_patch: np.ndarray, spec: dict[str, Any], mask: np.ndarray) -> list[np.ndarray]:
    """Slice one RF2 patch into flattened, optionally Gaussian-masked RF1 tiles."""
    rf1_patches = []
    for patch_x in range(spec['rf1_layout_x']):
        for patch_y in range(spec['rf1_layout_y']):
            x_start = spec['rf1_offset_x'] * patch_x
            y_start = spec['rf1_offset_y'] * patch_y
            rf1_patch = rf2_patch[y_start:y_start + spec['rf1_y'], x_start:x_start + spec['rf1_x']].reshape(-1)
            if bool(spec['use_mask']):
                rf1_patch = rf1_patch * mask.reshape(-1)
            rf1_patches.append(rf1_patch)
    return rf1_patches


def _build_static_spcc_dataset_xy(spec: dict[str, Any], script_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build tiled sPCC (X, Y) arrays from raw PNGs per a dataset spec.

    Returns:
        Tuple of X with shape (N_inputs, N_tiles, flattened_RF1) and Y with
        shape (N_inputs, N_classes).
    """
    raw_dir = script_dir / 'data' / 'raw' / spec['raw_dir']
    images = _load_static_png_images(str(raw_dir))
    filtered_images = np.asarray([_apply_image_filter(image, spec) for image in images], dtype=np.float32)
    rf2_patches, labels = _extract_rf2_patches(filtered_images, spec)
    mask = _create_gauss_mask(
        sigma=float(spec['gauss_mask_sigma']),
        width=int(spec['rf1_x']),
        height=int(spec['rf1_y']),
    )
    x_data = np.asarray(
        [_rf1_patches_from_rf2_patch(rf2_patch, spec, mask) for rf2_patch in rf2_patches],
        dtype=np.float32,
    )
    y_data = np.asarray(labels, dtype=np.float32)
    return x_data, y_data


def _rf_metadata_path(script_dir: Path, dataset_name: str) -> Path:
    """Return the RF metadata file path for a dataset, creating data/metadata/ if needed."""
    metadata_dir = script_dir / 'data' / 'metadata'
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / f'{dataset_name}_rf_metadata.txt'


def _write_spcc_rf_metadata(script_dir: Path, spec: dict[str, Any]) -> Path:
    """Write receptive-field layout metadata for a tiled sPCC dataset and return its path."""
    metadata = {
        'dataset_name': spec['dataset_name'],
        'rf1_x': int(spec['rf1_x']),
        'rf1_y': int(spec['rf1_y']),
        'rf1_offset_x': int(spec['rf1_offset_x']),
        'rf1_offset_y': int(spec['rf1_offset_y']),
        'rf1_layout_x': int(spec['rf1_layout_x']),
        'rf1_layout_y': int(spec['rf1_layout_y']),
        'ntiles_per_input': int(spec['rf1_layout_x']) * int(spec['rf1_layout_y']),
        'flat_input': True,
        'use_mask': bool(spec['use_mask']),
        'gauss_mask_sigma': float(spec['gauss_mask_sigma']),
        'image_filter': spec['image_filter'],
        'raw_dir': spec['raw_dir'],
    }
    metadata_path = _rf_metadata_path(script_dir, spec['dataset_name'])
    metadata_lines = [
        '# SPCC receptive-field overlap metadata',
        '# Saved alongside tiled datasets for input-space reconstruction during plotting.',
    ]
    for key, value in metadata.items():
        metadata_lines.append(f'{key}={value!r}')
    metadata_path.write_text('\n'.join(metadata_lines) + '\n', encoding='utf-8')
    return metadata_path


def _preprocess_static_spcc_dataset(dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Build, pickle, and describe a static sPCC dataset named in SPCC_DATASET_SPECS."""
    spec = dict(SPCC_DATASET_SPECS[dataset_name])
    script_dir = Path(__file__).resolve().parent
    x_data, y_data = _build_static_spcc_dataset_xy(spec, script_dir)
    out_path = script_dir / 'data' / f'{dataset_name}.pydb'
    with out_path.open('wb') as out_file:
        pickle.dump((x_data, y_data), out_file)
    metadata_path = _write_spcc_rf_metadata(script_dir, spec)

    print(f"X shape: {x_data.shape}  (N_inputs, N_tiles, flattened_RF1)")
    print(f"Y shape: {y_data.shape}  (N_inputs, N_classes)")
    print(f"Saved: {out_path}")
    print(f"Saved metadata: {metadata_path}")
    return x_data, y_data


def preprocess_raonaturalimages5_for_spcc() -> tuple[np.ndarray, np.ndarray]:
    """Preprocess raw 5Nat images into the frozen sPCC config's X/Y format."""
    return _preprocess_static_spcc_dataset('Rogers_2026_raonaturalimages5')


def preprocess_trace212_for_spcc() -> tuple[np.ndarray, np.ndarray]:
    """Preprocess raw trace212 images into the frozen sPCC config's X/Y format."""
    return _preprocess_static_spcc_dataset('Rogers_2026_trace212')


if __name__ == "__main__":
    if "--cvcv12" in sys.argv:
        preprocess_cvcv12_for_rpcc()
    elif "--tom3phon3let_16set" in sys.argv or "tom3phon3let_16set" in sys.argv:
        preprocess_tom3phon3let_for_rpcc(lexicon_choice=TOM3PHON3LET_16SET_WORDS)
    elif "--tom3phon3let_4set" in sys.argv or "tom3phon3let_4set" in sys.argv:
        preprocess_tom3phon3let_for_rpcc(lexicon_choice=TOM3PHON3LET_4SET_WORDS)
    elif "--raonaturalimages5" in sys.argv:
        preprocess_raonaturalimages5_for_spcc()
    elif "--trace212" in sys.argv:
        preprocess_trace212_for_spcc()
    else:
        print(
            "Usage: python data.py --cvcv12 | --tom3phon3let_16set | --tom3phon3let_4set | "
            "--raonaturalimages5 | --trace212"
        )
        sys.exit(1)
