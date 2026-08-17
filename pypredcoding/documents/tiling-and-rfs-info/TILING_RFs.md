## Tiling
SPCC tiling means one input is represented as receptive-field patches instead of a single untiled image array.

Support: the model currently supports four static input modes (with size constraints in parens): tiled flat `(n_tiles, patch_values)`, tiled expanded `(tile_rows, tile_cols, patch_values)`, untiled flat `(input_values,)`, and untiled expanded `(height, width)`.

Current use cases: `data.py` writes both `Rogers_2026_raonaturalimages5.pydb` and `Rogers_2026_trace212.pydb` in tiled-flat form, with `X shape: (N_inputs, N_tiles, flattened_RF1)`. Their frozen configs then pair that data with `ntiles_per_input` set and `flat_input=True`.

Limitations: the current shipped sPCC examples are tiled-flat only; other static modes are supported by the model contract but are not illustrated by frozen baseline datasets here. Input shapes outside those four modes are not supported.

## Receptive Fields (plots)
SPCC receptive fields are saved only for static models when `plot_receptive_fields=True` and the normal plot-save path runs (checkpoints or finished models). They print reconstructed images from weights for all extant layers.

Support: the RF panels are computed online from the current model and the configured training dataset, using `X` only from either `(X, Y)` or `X`-only `.pydb` files. The renderer supports exactly four static input modes: tiled flat, tiled expanded, untiled flat, and untiled expanded. ..data/metadata/DATASETNAME_rf_metadata.txt files containing the original tiling constraints are needed to reconstruct (a) raw, (b) receptive field panels in original image-space dimensions. In the absence of correct metadata, a tiled dataset producing receptive fields will simply plot panels in a tiled, mosaic form (i.e., where tiles do not overlap).

Orchestration: calculated online from current model inst., assoc. dataset, and optional metadata. Only artifacts that remain after training: epoch and experiment-indexed receptive_field/ dirs with named PNGs inside.

Current use cases: the shipped frozen SPCC baselines exercise tiled-flat inputs, including `Rogers_2026_raonaturalimages5.pydb` and `Rogers_2026_trace212.pydb`.

Limitations: recurrent models reject this flag, unsupported input-mode/shape combinations raise, and RF images are display projections rather than a separate diagnostics artifact format.