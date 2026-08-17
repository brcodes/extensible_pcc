# test_model_construction.py — Coverage README

Run from the `pypredcoding` root: `python tests/test_model_construction.py` (prints `status: PASS` on success).

## Summary

- **Architecture & construction soundness** — factory vs direct construction parity, valid instantiation across the full feature matrix (model type, depth, input type, connection, architecture, classification, activation, priors), and correct state/weight shapes.
- **Update/cost call-chain ("connector") tests** — every valid feature combination drives the exact update-dispatch → representation/classification-cost → classify → eval call chain that `train()`/`evaluate()` consume, asserting finiteness, shape invariants, eval-path weight freezing, and seeded determinism.
- **Training & evaluation plumbing** — miniature end-to-end runs per model type: train completion, checkpoint saving, artifact existence, pickle reload fidelity, resume-from-checkpoint, and run-to-run determinism.
- **Parameterization & config validation** — required keys, key-policy rules ('o' keys, classification prerequisites, plotting flags, rate schedules), and rejection of invalid architecture/feature combinations with the intended error messages.

## Construction & factory tests

- `create_model` returns the correct concrete class (`StaticPCC` / `RecurrentPCC`) for both frozen baseline configs, and factory vs direct construction produce identical initial `U` (and `V`) weights under the same seed.
- Static 1-layer models construct from both frozen sPCC baselines (trimmed layer dicts); `all_lyr_sizes`, `r[1]`, `U[1]` present and correct.
- Static c2 permits `top_lyr_size != num_classes`; the c2 top-down signal shape matches the top layer.
- 2-layer reshaped static: `U[2]` flattens the expanded `r[1]` (`(prod(r1), top)`) and `U2r2_e1l_Li` reproduces `r[1]`'s shape.
- Recurrent 1-layer model constructs with complete state dicts (`r/U/V`, `rhat/rbar`, `Uhat/Ubar`, `Vhat/Vbar`) plus the `'c'` context states.
- Base class `PredictiveCodingClassifier` is not directly instantiable.

## No-classification (classif_method=None) behavior

- Static and recurrent NC models allocate no `Jc`/`accuracy` histories; `evaluate` returns `None` (recurrent detail dict returns absent accuracy/predictions).
- NC diagnostics save and plot successfully; expected plot PNG artifacts exist on disk.
- Receptive-field plots generate for the natural-images baseline, including an X-only (no labels) dataset variant; tiled-flat RF rendering rejects wrongly shaped arrays.

## Feature-matrix connector tests

Each valid combination runs the exact call chain training/evaluation consume: construct → prior reset → weight-bearing update dispatch → `rep_cost`/`classif_cost` → no-weight (eval) updates → `classify`/`evaluate`.

Feature axes covered:

- **A. Model type** — static (sPCC) and recurrent (rPCC).
- **B. Depth** — n = 1, 2, 3, 4 (n≥3/4 exercises the generic middle-layer loops).
- **C. Input type (static)** — tiled flat, tiled nonflat, untiled flat, untiled nonflat; recurrent uses vector-timestep input.
- **D. First-to-second connection** — structured and reshaped (reshaped incl. n=2 top-layer flattening and n=3 hidden flattening, with expected `U[2]` shapes).
- **E. Architecture** — `expand_first_lyr` and `flat_hidden_lyrs` (with expected `r[1]` shapes per input type).
- **F. Classification** — c1, c2, and None for both model types.
- **G. Activation** — linear and tanh.
- **H. Prior costs** — gaussian and sparse_kurtotic (static only; rPCC prior costs must be None).
- **Update methods** — all three static dispatch keys (`rW_instdeltas_niters`, `r_niters_W`, `r_eq_W`) and the recurrent `rW_seq_niters`.
- **Softmax** — recurrent stable and normal variants.

Per-connector assertions:

- Correct concrete class; `all_lyr_sizes` length; `U[1]`/`r[1]` shape invariants; per-combo expected `r[1]`/`U[2]` shapes; `U['o']` shape for c2; recurrent `rbar/rhat/Uhat/Vhat` + `'c'` completeness.
- All post-update states and costs are finite; accuracy in [0, 1]; static classify returns 0/1; recurrent evaluate returns one prediction per input.
- Eval (no-weight) paths leave weights bit-identical (`U` static; `Uhat`/`Vhat` recurrent).
- Determinism: each connector runs twice under a fixed seed and full state+cost fingerprints must be exactly equal.

## Invalid-combination rejection

- Static n=1 + tiled + `expand_first_lyr` + c1/c2 raises the non-vectorial top-layer error.
- `reshaped` connection without tiled flat input raises.
- rPCC with any prior cost set raises the prior-cost stub error.
- rPCC with `architecture='expand_first_lyr'` raises (rPCC requires `flat_hidden_lyrs`).
- Unsupported `model_type` raises listing supported types.

## Training & evaluation plumbing (end-to-end, per model type)

Miniature runs (2 epochs, 2 inputs, temp dirs) on representative configs — static: tiled flat / expand / reshaped / c1; recurrent: 2-layer / c1.

- Training completes; `Jr` (and `Jc`/`accuracy` when classifying) histories are the right length and finite.
- Checkpoint model saved at epoch 1; final model saved in the models dir; diagnostics artifact exists and contains `Jr`/`Jc`/`accuracy`.
- Pickle reload of the final model matches in-memory weights exactly and evaluates identically (accuracy and, recurrent, predictions).
- Resume: the epoch-1 checkpoint reloads and continues training to completion, saving a final model.
- Determinism: two identical pipeline runs under fixed seeds reproduce histories and final weights exactly.

## Config-load & key-policy validation

- Required-key enforcement: missing static (`ssq`) / recurrent (`ssqr`) controls and missing `flat_input` are rejected at model construction; non-bool `flat_input` rejected.
- Baseline configs pass with the counterpart model type's keys removed (static keys stripped for rPCC, recurrent keys stripped for sPCC).
- 'o'-key policy: `kr`/`ssq`/`ssqr`/`ssqV` must not contain `'o'`; c2 requires `kU['o']` (and static `lam['o']`).
- c1 requires `num_classes == top_lyr_size`; `classif_method` must be `'c1'`, `'c2'`, or `None` (no `'nc'` alias).
- Plot flags: `plot_receptive_fields` is static-only, requires `plot_train=True`, must be bool/dict, and `random_sample` must be a positive integer; valid modes (`None`, `'all'`, `{'random_sample': N}`) load and construct. `plot_softmaxed_activations` is recurrent-only, must be explicit, and requires `plot_train=True`.
- Rate schedules: `'o'` key only allowed with c2; a valid c2 schedule loads and constructs.
- Untiled static configs (flat and nonflat) load and construct from the baseline via config-text mutation.
