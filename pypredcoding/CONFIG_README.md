# Extensible rPCC/sPCC Config README
### pypredcoding/
- ./config/config_....txt 

Use guide for extensible r/sPCC config files and parameters. Some examples drawn from validated Rogers et al. 2026 baselines:

- `config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt`
- `config_sPCC_Rogers_2026_raonaturalimages5_frozen_baseline.txt`
- `config_sPCC_Rogers_2026_trace212_frozen_baseline.txt`

Args are parsed as Python literals. Args must be in `key=value` form, and take up only a single line. Comments (# delineated) are ignored. Malformed syntax fails before model construction.

- `config_active.txt` is meant for experimentation; name it whatever you want.

## CONFIG EXAMPLE (RPCC CVCV12 FROZEN BASELINE)
```
# ------------------------------------------------------------------
#      TRAINING CONFIG FOR STATIC or RECURRENT PREDICTIVE CODING CLASSIFIER
#      1. SHARED PARAMS 2. RPCC-SPECIFIC 3. SPCC-SPECIFIC
# ------------------------------------------------------------------

# ----------------------
#       SHARED
# ----------------------

# MODEL TYPE
model_type='recurrent'

# EXPERIMENT METADATA
notes='Rogers/Li 2026 rPCC baseline- 12 CVCV words'

# INPUTS
input_shape=(21,)
num_inps=12
num_classes=12
dataset_train='Rogers_2026_cvcv12.pydb'

# TRAINING / HYPERPARAMETERS
epoch_n=5000
seed_init=1
seed_shuffle=1
batch_size=1
load_checkpoint=None
save_checkpoint={'save_every':100}
plot_train=True
plot_softmaxed_activations=True
kr={1:0.1, 2:5.0}
kU={1:0.1, 2:0.1}
rate_schedule=None

# ARCHITECTURE
num_layers=2
hidden_lyr_sizes=[10]
top_lyr_size=12
classif_method='c1'
activ_func='linear'
update_method={'rW_seq_niters': 1}
architecture='flat_hidden_lyrs'
first_to_second_lyr_connection_architecture='structured'
r_reset_mode='reset_from_single_sample'
r_prior_dist=('pseudo_sparse_kurtotic',)
r_prior_cost=None
r_prior_cost_denominator=None
U_prior_dist=('gaussian', 0.0, 0.1)
U_prior_cost=None
U_prior_cost_denominator=None
softmax_type='stable'
softmax_k=1

# ----------------------
#      RPCC ONLY
# ----------------------

# INPUTS
num_ts=98

# TRAINING / HYPERPARAMETERS
kV={1:0.01, 2:0.01}
ssqr={0:10, 1:10, 2:5}
ssqV={1:1, 2:1}

#ARCHITECTURE
rc_topdown_cost_denominator=2
softmax_k_eval=20

# ----------------------
#      SPCC ONLY
# ----------------------

# INPUTS
ntiles_per_input=None
flat_input=None

# TRAINING / HYPERPARAMETERS
ssq=None
alph=None
lam=None
plot_receptive_fields=False
```

# PARAMETERS
## SHARED (R & SPCC)
### MODEL TYPE
### `model_type`

- Function: Selects the model family.
- Arguments:
  - Type/options: string.
  - Description:
      - `'recurrent'`: builds `RecurrentPCC`; uses sequence/timestep state, recurrent `V` weights, rPCC validation, and recurrent plotting/evaluation behavior.
      - `'static'`: builds `StaticPCC`; uses static input state, sPCC tiling/receptive-field options, and static validation.
  - Examples:
      - `model_type='recurrent'`: rPCC CVCV12 baseline.
      - `model_type='static'`: sPCC natural-image and trace baselines.
- Notes:
  - Unknown values fail at factory construction because no model family can be selected.
  - Recurrent models reject static-only receptive-field plotting and static-only architectures.

### EXPERIMENT METADATA
### `notes`

- Function: Labels the run intent.
- Arguments:
  - Type/options: string.
  - Description:
      - Any free-text note; stored with params/logs so a saved run can be identified later.
  - Examples:
      - `notes='Rogers/Li 2026 rPCC baseline- 12 CVCV words'`: rPCC baseline note.
      - `notes='Rogers/Li 2026 sPCC baseline- 5 natural images'`: sPCC natural-image baseline note.
- Notes:
  - Required by both model classes; omitting it raises `Config missing required keys: notes`.

### INPUTS
### `input_shape`

- Function: Defines one input sample's model-facing shape.
- Arguments:
  - Type/options: tuple of positive dimensions matching one `X` sample, excluding sample index and recurrent timestep axis.
  - Description:
      - rPCC: shape of the acoustic feature vector at one timestep; the recurrent time axis is controlled separately by `num_ts`.
      - sPCC tiled-flat: `(ntiles_per_input, flattened_tile_length)`; examples are image/trace receptive-field tiles.
      - sPCC untiled modes: tuple describing the full static input shape.
  - Examples:
      - `input_shape=(21,)`: rPCC CVCV12 acoustic vector per timestep.
      - `input_shape=(225,256)`: sPCC 5-natural-image tiled-flat inputs.
      - `input_shape=(16,864)`: sPCC trace212 tiled-flat inputs.
- Notes:
  - Must match the loaded dataset; mismatches usually fail later in NumPy dot/einsum operations.
  - Tiled-flat static baselines pair `input_shape=(ntiles_per_input, flattened_tile_length)` with `ntiles_per_input` and `flat_input=True`.

### `num_inps`

- Function: States dataset sample count.
- Arguments:
  - Type/options: integer.
  - Description:
      - Number of input samples the training/evaluation loops expect to iterate over.
  - Examples:
      - `num_inps=12`: rPCC CVCV12 baseline.
      - `num_inps=5`: sPCC natural-image baseline.
      - `num_inps=212`: sPCC trace baseline.
- Notes:
  - Should match `X.shape[0]`; mismatches can cause skipped samples, index errors, or misleading reporting.

### `num_classes`

- Function: States output class count.
- Arguments:
  - Type/options: integer.
  - Description:
      - Label-vector width and classifier output count.
      - For `classif_method='c1'`, this is also the required top-layer size.
      - For `classif_method='c2'`, this is the output-readout class/logit size, while `top_lyr_size` can be latent-sized.
  - Examples:
      - `num_classes=12`: rPCC CVCV12 baseline.
      - `num_classes=5`: sPCC natural-image baseline.
      - `num_classes=212`: sPCC trace baseline.
- Notes:
  - With `classif_method='c1'`, must equal `top_lyr_size`; otherwise config parsing raises `num_classes must equal top_lyr_size when classif_method='c1'.`
  - With `classif_method='c2'`, `top_lyr_size` may be latent-sized, but `kU['o']` is required and static `lam['o']` is also required.

### `dataset_train`

- Function: Selects the training dataset file.
- Arguments:
  - Type/options: string filename with suffix, or bare dataset stem resolved under `pypredcoding/data/` with `.pydb`/`.pkl` fallbacks.
  - Description:
      - Dataset pickle must provide training inputs and labels in shapes compatible with the config.
      - Receptive-field plotting can also consume `X`-only pickles for static RF rendering.
  - Examples:
      - `dataset_train='Rogers_2026_cvcv12.pydb'`: rPCC CVCV12 data.
      - `dataset_train='Rogers_2026_raonaturalimages5.pydb'`: sPCC natural-image data.
      - `dataset_train='Rogers_2026_trace212.pydb'`: sPCC trace data.
- Notes:
  - Missing files raise `Dataset file not found: ... Expected a premade pickle named one of: ...`
  - Dataset shape must agree with `num_inps`, `input_shape`, `num_classes`, and, for rPCC, `num_ts`.

### TRAINING / HYPERPARAMETERS
### `epoch_n`

- Function: Sets training duration.
- Arguments:
  - Type/options: integer target epoch count.
  - Description:
      - Number of full passes through the configured training set.
      - Also sizes epoch-indexed diagnostics histories.
  - Examples:
      - `epoch_n=5000`: rPCC CVCV12 baseline.
      - `epoch_n=500`: sPCC natural-image baseline.
      - `epoch_n=300`: sPCC trace baseline.
- Notes:
  - When resuming from checkpoint, `epoch_n` must be at least the loaded checkpoint epoch; otherwise training raises a message ending with `Boost epoch_n in config greater than ... before training.`

### `seed_init`

- Function: Controls initialization reproducibility.
- Arguments:
  - Type/options: `None` or integer.
  - Description:
      - Integer: seeds NumPy before fresh model construction so initial `r`, `U`, and `V` states are reproducible.
      - `None`: leaves initialization randomness unseeded by this key.
  - Examples:
      - `seed_init=1`: validated frozen baselines.
      - `seed_init=None`: unseeded initialization.
- Notes:
  - Booleans and non-integers raise `seed must be None or an integer.`
  - Console output when set: `Initialization seed set from config: 1`.
  - Loading a checkpoint restores saved weights/states; this seed does not resample a loaded checkpoint.

### `seed_shuffle`

- Function: Controls training-order reproducibility.
- Arguments:
  - Type/options: `None` or integer.
  - Description:
      - Integer: seeds NumPy immediately before training, controlling sample shuffle order.
      - `None`: leaves shuffle order unseeded by this key.
  - Examples:
      - `seed_shuffle=1`: validated frozen baselines.
      - `seed_shuffle=None`: unseeded shuffle.
- Notes:
  - Booleans and non-integers raise `seed must be None or an integer.`
  - Console output when set: `Training shuffle seed set from config: 1`.

### `batch_size`

- Function: Sets training batch size.
- Arguments:
  - Type/options: integer.
  - Description:
      - `1`: stochastic single-input updates; this is the validated and currently supported path.
      - `>1`: not yet supported; SGD only.
  - Examples:
      - `batch_size=1`: all frozen baselines.
- Notes:
  - Required by both model classes; omitting it raises `Config missing required keys: batch_size`.
  - The frozen baselines validate `batch_size=1`; other values should be smoke-tested before long runs.

### `load_checkpoint`

- Function: Selects fresh start versus checkpoint resume.
- Arguments:
  - Type/options: `None`, `-1`, non-negative integer epoch, or tuple `(epoch, 'exp_YYMMDD_HHMMSS')` where tuple epoch is `-1` or a positive integer.
  - Description:
      - `None`: instantiate a fresh model from configured priors.
      - `-1`: load the latest checkpoint from the relevant checkpoint directory.
      - Non-negative integer: load the checkpoint for that exact epoch.
      - `(epoch, 'exp_YYMMDD_HHMMSS')`: load a checkpoint from another experiment directory.
  - Examples:
      - `load_checkpoint=None`: frozen baselines; start fresh.
      - `load_checkpoint=-1`: resume from latest checkpoint.
      - `load_checkpoint=100`: resume from epoch 100 checkpoint.
      - `load_checkpoint=(-1, 'exp_260804_095351')`: resume latest checkpoint from a specific experiment.
- Notes:
  - Invalid forms raise `load_checkpoint must be None, -1, a non-negative integer epoch, or a tuple like (epoch, 'exp_YYMMDD_HHMMSS').`
  - Tuple epoch `0`, bools, non-integers, or negative values other than `-1` raise `load_checkpoint tuple epoch must be -1 or a positive integer.`
  - Empty/non-string tuple experiment names raise `load_checkpoint tuple experiment name must be a non-empty string like 'exp_YYMMDD_HHMMSS'.`
  - Missing checkpoint files raise `No matching checkpoint found`, `No valid checkpoint filenames found`, or `No checkpoint found for requested epoch N`.
  - Console output includes `Loading latest checkpoint`, `Loading checkpoint at epoch N`, and `Loaded checkpoint: ...`.

### `save_checkpoint`

- Function: Sets checkpoint save policy.
- Arguments:
  - Type/options: dictionary; `{'save_every': N}`, `{'fraction': x}`, `{'fraction': (num, denom)}`, or `{}`.
  - Description:
      - `{'save_every': N}`: save every `N` epochs.
      - `{'fraction': x}`: save every `ceil(epoch_n * x)` epochs.
      - `{'fraction': (num, denom)}`: save every `ceil(epoch_n * num / denom)` epochs.
      - `{}`: save no intermediate checkpoints.
  - Examples:
      - `save_checkpoint={'save_every':10}`: frozen baselines.
      - `save_checkpoint={}`: no intermediate checkpointing.
      - `save_checkpoint={'fraction':0.1}`: checkpoint at roughly 10% intervals.
      - `save_checkpoint={'fraction':(1,4)}`: checkpoint at roughly quarter intervals.
- Notes:
  - Active checkpointing logs `Checkpoint method: ... saving every N`.
  - `plot_train=True` plots at saved checkpoints and final output; with `{}`, only final plotting runs.
  - If a `fraction` policy rounds to `0` for a very small `epoch_n`, checkpoint modulo logic can fail; use `{'save_every':1}` for tiny smoke runs.

### `plot_train`

- Function: Toggles training diagnostics plots.
- Arguments:
  - Type/options: boolean.
  - Description:
      - `True`: save diagnostics/plots through the plot-save path.
      - `False`: skip plot artifact generation.
  - Examples:
      - `plot_train=True`: frozen baselines.
      - `plot_train=False`: fast/headless smoke runs.
- Notes:
  - Enabled `plot_receptive_fields` requires `plot_train=True`; otherwise parsing raises `plot_receptive_fields requires plot_train=True.`
  - `plot_softmaxed_activations=True` requires `plot_train=True`; otherwise parsing raises `plot_softmaxed_activations requires plot_train=True.`

### `plot_softmaxed_activations`

- Function: Toggles softmaxed activation diagnostics.
- Arguments:
  - Type/options: explicit boolean.
  - Description:
      - `True`: compute/save top-layer softmaxed activation diagnostics where supported.
      - `False`: skip the softmaxed activation panel and related extraction.
  - Examples:
      - `plot_softmaxed_activations=True`: rPCC CVCV12 baseline.
      - `plot_softmaxed_activations=False`: sPCC baselines.
- Notes:
  - Missing or non-boolean values raise `plot_softmaxed_activations must be explicitly set to True or False.`
  - Requires `plot_train=True`; otherwise raises `plot_softmaxed_activations requires plot_train=True.`
  - Static models reject `True` with `plot_softmaxed_activations is not supported for static models...` because current SPCC word datasets do not provide the lookup maps required for cohort/rhyme/unrelated activation summaries.

### `kr`

- Function: Sets representation learning rates.
- Arguments:
  - Type/options: dictionary with integer keys `1..num_layers`; no `'o'` key.
  - Description:
      - `kr[i]`: learning rate for representation/state updates at layer `i`.
      - Larger values make r-state corrections stronger; the effect depends on the corresponding error denominator.
  - Examples:
      - `kr={1:0.1, 2:5.0}`: rPCC baseline.
      - `kr={1:0.005, 2:0.005, 3:0.005}`: sPCC natural-image baseline.
      - `kr={1:0.001, 2:0.001, 3:0.001}`: sPCC trace baseline.
- Notes:
  - Missing integer keys raise `kr missing required key(s): [...]`.
  - An `'o'` key raises `kr must not contain an 'o' key.`

### `kU`

- Function: Sets generative-weight learning rates.
- Arguments:
  - Type/options: dictionary with integer keys `1..num_layers`; add `'o'` only for `classif_method='c2'`.
  - Description:
      - `kU[i]`: learning rate for generative weights `U[i]` at layer `i`.
      - `kU['o']`: c2 output-readout learning rate; only meaningful when `classif_method='c2'`.
  - Examples:
      - `kU={1:0.1, 2:0.1}`: rPCC c1 baseline.
      - `kU={1:0.01, 2:0.01, 3:0.01}`: sPCC natural-image baseline.
      - `kU={1:0.02, 2:0.94, 'o':0.02}`: c2-style output-readout example.
- Notes:
  - Missing integer keys raise `kU missing required key(s): [...]`.
  - `'o'` without `classif_method='c2'` raises `kU may contain an 'o' key only when classif_method='c2'.`
  - `classif_method='c2'` without `'o'` raises `kU must include an 'o' key when classif_method='c2'.`

### `rate_schedule`

- Function: Schedules learning-rate changes.
- Arguments:
  - Type/options: `None`, one schedule dictionary, or a list/tuple of schedule dictionaries.
  - Description:
      - `None`: keep all configured learning rates fixed.
      - `piecewise_linear`: interpolate one learning-rate entry across `(epoch, LR)` points; layer number/output key is stored in `key`.
      - `two_phase_linear_decay`: hold the base rate until `start_epoch`, linearly decay to `final_value` by `end_epoch`, then hold `final_value`.
      - List/tuple: apply multiple schedules simultaneously.
  - Examples:
      - `rate_schedule=None`: frozen baselines.
      - `rate_schedule={'type':'piecewise_linear','param':'kr','key':2,'points':[(0,0.125),(700,0.0125)]}`: schedule `kr[2]`.
      - `rate_schedule={'type':'two_phase_linear_decay','param':'kU','key':1,'start_epoch':10,'end_epoch':100,'final_value':0.001}`: schedule `kU[1]`.
- Notes:
  - Non-dict schedule entries raise `rate_schedule must be None, a schedule dictionary, or a list/tuple of schedule dictionaries.` or `rate_schedule iterable must contain only schedule dictionaries.`
  - Unsupported types raise `Unsupported rate_schedule type: ...`
  - `piecewise_linear` requires at least one point and strictly increasing epochs.
  - `two_phase_linear_decay` requires `end_epoch >= start_epoch`; otherwise raises `rate_schedule end_epoch must be greater than or equal to start_epoch.`
  - `param` must be one of `kr`, `kU`, or `kV`; otherwise training raises `Unsupported rate_schedule param: ...`
  - Schedule key `'o'` is only allowed when `classif_method='c2'`, and only for `param='kU'`.

### ARCHITECTURE
### `num_layers`

- Function: Sets model depth.
- Arguments:
  - Type/options: integer `>= 1`.
  - Description:
      - Counts non-input layers, including the top layer.
      - Determines required dictionary key ranges for per-layer parameters.
  - Examples:
      - `num_layers=2`: rPCC CVCV12 baseline.
      - `num_layers=3`: sPCC baselines.
- Notes:
  - Values `< 1` raise `num_layers must be at least 1.`
  - `hidden_lyr_sizes` must contain exactly `num_layers - 1` entries.

### `hidden_lyr_sizes`

- Function: Sets hidden-layer widths.
- Arguments:
  - Type/options: list of integers; length must be `num_layers - 1`.
  - Description:
      - Each list element gives one hidden layer size before the top layer.
      - Empty list is valid only for a one-layer model.
  - Examples:
      - `hidden_lyr_sizes=[10]`: two-layer rPCC baseline.
      - `hidden_lyr_sizes=[32,128]`: three-layer sPCC baselines.
      - `hidden_lyr_sizes=[]`: one-layer model.
- Notes:
  - Wrong length raises `hidden_lyr_sizes must contain N entries for num_layers=M, got K.`

### `top_lyr_size`

- Function: Sets top-layer width.
- Arguments:
  - Type/options: integer.
  - Description:
      - Size of `r[num_layers]`, the top representation layer.
      - In `c1`, this layer directly represents class logits and must equal `num_classes`.
      - In `c2`, this layer can be arbitrary/latent-sized because `U['o']` maps it into class-logit space.
  - Examples:
      - `top_lyr_size=12`: rPCC CVCV12 baseline.
      - `top_lyr_size=5`: sPCC natural-image baseline.
      - `top_lyr_size=212`: sPCC trace baseline.
- Notes:
  - With `classif_method='c1'`, must equal `num_classes`; otherwise parsing raises `num_classes must equal top_lyr_size when classif_method='c1'.`
  - With `classif_method='c2'`, this can be a latent size, but output-readout learning keys are required.

### `classif_method`

- Function: Selects classification mode.
- Arguments:
  - Type/options: `'c1'`, `'c2'`, or `None`.
  - Description:
      - `'c1'`: direct classification; top-layer `r` expresses `num_classes` logits.
      - `'c2'`: learned output readout; top-layer `r` may be arbitrary-sized and `U['o']` maps it into `num_classes` logits.
      - `None`: no classification; unsupervised inter-layer learning remains through input reconstruction and prior terms.
  - Examples:
      - `classif_method='c1'`: frozen baselines.
      - `classif_method='c2'`: learned output-readout experiments.
      - `classif_method=None`: no classification cost/metrics.
- Notes:
  - Other values raise `classif_method must be 'c1', 'c2', or None.`
  - `c1` requires `num_classes == top_lyr_size`.
  - `c2` requires `kU['o']`; static `c2` also requires `lam['o']`.
  - `None` disables classification metrics; evaluation returns absent accuracy/predictions.

### `activ_func`

- Function: Selects activation nonlinearity.
- Arguments:
  - Type/options: `'linear'` or `'tanh'`.
  - Description:
      - `'linear'`: identity transform for generative predictions; validated baseline path.
      - `'tanh'`: hyperbolic tangent transform with derivative terms.
  - Examples:
      - `activ_func='linear'`: frozen baselines.
      - `activ_func='tanh'`: nonlinear experiment.
- Notes:
  - Applies to every generative `U.r` prediction (bottom-up errors, top-down predictions,
    rep costs, receptive-field reconstruction) in both sPCC and rPCC, with the derivative
    `F` modulating the corresponding r/U gradient terms.
  - Never applies to `V.r` temporal predictions (always linear) or to classification
    logits (`softmax(r_n)`, `softmax(Uo.r)`).
  - Unknown values fail through activation-function lookup.
  - The validated baselines use `linear` (identity — numerically exact vs. frozen models);
    `tanh` should be smoke-tested before long runs.

### `update_method`

- Function: Selects update schedule.
- Arguments:
  - Type/options: single-key dictionary.
  - Description:
      - `rW_seq_niters`: rPCC only; per timestep, runs r updates then each weight update in
        sequence (weights see the freshly corrected r states), `niters` times. Baseline uses `1`.
      - `rW_instdeltas_niters`: sPCC only; instantaneous-delta method — computes r deltas and
        weight deltas from the pre-update state, then applies both sets of deltas. sPCC
        baselines use `30` iterations per input.
      - `r_niters_W`: currently sPCC only; runs r updates `niters` times first, then runs each weight update once. Weights see the post-r-updated state, not the original pre-r state.
      - `r_eq_W`: currently sPCC only; runs r updates until relative r-state change is below the stop criterion (% units), then runs each weight update once. Weights see the post-r-updated state, not the original pre-r state.
  - Examples:
      - `update_method={'rW_seq_niters': 1}`: rPCC baseline.
      - `update_method={'rW_instdeltas_niters':30}`: sPCC baselines.
      - `update_method={'r_eq_W':5}`: stop when relative r-state norm change is below 5%.
- Notes:
  - Unsupported keys fail when training looks up the update method.
  - For rPCC specifically, only `rW_seq_niters` is supported; `r_niters_W` and `r_eq_W` are not currently supported despite being present in the update-method dictionary.
  - `rW_instdeltas_niters` is not registered for rPCC; recurrent configs must use `rW_seq_niters`.
  - Multiple keys are not meaningful because training uses the first dictionary key.

### `architecture`

- Function: Selects layer-shape architecture.
- Arguments:
  - Type/options: `'flat_hidden_lyrs'` or `'expand_first_lyr'`.
  - Description:
      - `'flat_hidden_lyrs'`: keep hidden layers flat; required for current rPCC.
      - `'expand_first_lyr'`: expand first static layer over input tiles/spatial positions to enable receptive-field-style representations; more biologically plausible for images, possibly speech, but not yet supported for rPCC.
  - Examples:
      - `architecture='flat_hidden_lyrs'`: rPCC baseline.
      - `architecture='expand_first_lyr'`: sPCC baselines.
- Notes:
  - Unsupported values raise `Unsupported architecture: ...`
  - rPCC requires `flat_hidden_lyrs`; otherwise raises `rPCC requires architecture='flat_hidden_lyrs'.`
  - `flat_hidden_lyrs` requires `first_to_second_lyr_connection_architecture='structured'`.
  - Static `num_layers == 1` plus tiled input plus `expand_first_lyr` plus `c1`/`c2` raises a top-layer non-vectorial classifier error; use `num_layers >= 2`, untiled input, or `flat_hidden_lyrs`.

### `first_to_second_lyr_connection_architecture`

- Function: Selects first-to-second connection shape.
- Arguments:
  - Type/options: `'structured'` or `'reshaped'`.
  - Description:
      - `'structured'`: preserve first-layer structure when connecting `r1` to `r2`; required with `flat_hidden_lyrs`.
      - `'reshaped'`: flatten expanded first-layer submodules before `U2` operations; exists to preserve unique sPCC training plumbing from the Rogers et al. 2026 legacy repo.
  - Examples:
      - `first_to_second_lyr_connection_architecture='structured'`: rPCC baseline and flat-hidden path.
      - `first_to_second_lyr_connection_architecture='reshaped'`: Li-style tiled-flat sPCC baselines.
- Notes:
  - Unknown values raise `first_to_second_lyr_connection_architecture must be one of {'structured', 'reshaped'}`.
  - `flat_hidden_lyrs` with `'reshaped'` raises `flat_hidden_lyrs requires first_to_second_lyr_connection_architecture='structured'.`
  - `'reshaped'` requires tiled-flat static input; otherwise raises `first_to_second_lyr_connection_architecture='reshaped' requires ntiles_per_input to be set and flat_input=True.`

### `r_reset_mode`

- Function: Selects r-state reset policy.
- Arguments:
  - Type/options: `'reset_from_single_sample'` or `'reset_with_continual_resampling'`.
  - Description:
      - `'reset_from_single_sample'`: sample/copy a fixed prior state for resets; validated baseline mode.
      - `'reset_with_continual_resampling'`: resample from the r prior when resetting states.
  - Examples:
      - `r_reset_mode='reset_from_single_sample'`: frozen baselines.
      - `r_reset_mode='reset_with_continual_resampling'`: stochastic reset experiment.
- Notes:
  - Unsupported values raise `Unsupported r_reset_mode '...'. Expected one of ...`
  - Training logs the chosen mode as `r_reset_mode: ...`.

### `r_prior_dist`

- Function: Selects r-state prior distribution.
- Arguments:
  - Type/options: tuple; supported names include `'gaussian'`, `'sparse_kurtotic'`, `'pseudo_sparse_kurtotic'`, and `'uniform_random'`.
  - Description:
      - `('pseudo_sparse_kurtotic',)`: zero-like pseudo-sparse prior used by the frozen baselines.
      - `('gaussian', mean, scale)`: Gaussian random prior.
      - `('sparse_kurtotic', mean, scale)`: sparse/Laplace-like random prior.
      - `('uniform_random', shift)`: uniform random prior shifted by `shift`.
  - Examples:
      - `r_prior_dist=('pseudo_sparse_kurtotic',)`: frozen baselines.
      - `r_prior_dist=('gaussian', 0.0, 0.1)`: Gaussian r prior.
      - `r_prior_dist=('sparse_kurtotic', 0.0, 0.1)`: sparse r prior.
      - `r_prior_dist=('uniform_random', -0.5)`: shifted uniform r prior.
- Notes:
  - Unknown names fail through prior sampler lookup.
  - Malformed named tuple strings raise `Unsupported prior dist tuple-string format: ...`
  - `pseudo_sparse_kurtotic` takes no numeric arguments in current code.

### `r_prior_cost`

- Function: Selects r prior cost.
- Arguments:
  - Type/options: for sPCC, `'gaussian'` or `'sparse_kurtotic'`; for rPCC, `None`.
  - Description:
      - `None`: no r prior cost driver; required for rPCC.
      - `'gaussian'`: Gaussian r prior cost/gradient for static updates.
      - `'sparse_kurtotic'`: sparse r prior cost/gradient for static updates.
  - Examples:
      - `r_prior_cost=None`: rPCC baseline.
      - `r_prior_cost='gaussian'`: sPCC natural-image baseline.
      - `r_prior_cost='sparse_kurtotic'`: sPCC trace baseline.
- Notes:
  - rPCC with non-`None` raises `rPCC has no prior cost drivers in its update terms...`
  - Unknown static cost names fail through prior-cost lookup.

### `r_prior_cost_denominator`

- Function: Scales r prior cost.
- Arguments:
  - Type/options: numeric denominator for sPCC; `None` for rPCC.
  - Description:
      - `None`: no denominator because rPCC has no r prior cost driver.
      - Numeric value: denominator applied to static r prior gradients; smaller values strengthen prior pressure.
  - Examples:
      - `r_prior_cost_denominator=None`: rPCC baseline.
      - `r_prior_cost_denominator=2`: sPCC baselines.
- Notes:
  - rPCC with non-`None` raises `rPCC has no prior cost drivers in its update terms..., so no denominators necessary.`
  - Very small values strengthen prior pressure and can dominate representation learning.

### `U_prior_dist`

- Function: Selects U/V prior distribution.
- Arguments:
  - Type/options: same tuple family as `r_prior_dist`.
  - Description:
      - `('gaussian', mean, scale)`: Gaussian generative-weight prior; rPCC baseline uses this for `U`, and recurrent transition weights `V` use the same configured prior distribution.
      - `('uniform_random', shift)`: shifted uniform weight prior; sPCC baselines use this.
      - `('sparse_kurtotic', mean, scale)`: sparse/Laplace-like weight prior.
  - Examples:
      - `U_prior_dist=('gaussian', 0.0, 0.1)`: rPCC baseline.
      - `U_prior_dist=('uniform_random', -0.5)`: sPCC baselines.
      - `U_prior_dist=('sparse_kurtotic', 0.0, 0.1)`: sparse U prior.
- Notes:
  - Prior distribution affects initialization even when recurrent prior costs are disabled.
  - Unknown names fail through prior sampler lookup.

### `U_prior_cost`

- Function: Selects U prior cost.
- Arguments:
  - Type/options: for sPCC, `'gaussian'` or `'sparse_kurtotic'`; for rPCC, `None`.
  - Description:
      - `None`: no U prior cost driver; required for rPCC.
      - `'gaussian'`: Gaussian U prior cost/gradient for static updates.
      - `'sparse_kurtotic'`: sparse U prior cost/gradient for static updates.
  - Examples:
      - `U_prior_cost=None`: rPCC baseline.
      - `U_prior_cost='gaussian'`: sPCC natural-image baseline.
      - `U_prior_cost='sparse_kurtotic'`: sPCC trace baseline.
- Notes:
  - rPCC with non-`None` raises `rPCC has no prior cost drivers in its update terms...`
  - Static `c2` also requires `lam['o']` for output-readout prior scaling.

### `U_prior_cost_denominator`

- Function: Scales U prior cost.
- Arguments:
  - Type/options: numeric denominator for sPCC; `None` for rPCC.
  - Description:
      - `None`: no denominator because rPCC has no U prior cost driver.
      - Numeric value: denominator applied to static U prior gradients; smaller values strengthen weight prior pressure.
  - Examples:
      - `U_prior_cost_denominator=None`: rPCC baseline.
      - `U_prior_cost_denominator=2`: sPCC baselines.
- Notes:
  - rPCC with non-`None` raises `rPCC has no prior cost drivers in its update terms..., so no denominators necessary.`
  - Very small values strengthen weight prior pressure.

### `softmax_type`

- Function: Selects softmax implementation.
- Arguments:
  - Type/options: `'normal'` or `'stable'`.
  - Description:
      - `'normal'`: plain exponentiate-and-normalize softmax.
      - `'stable'`: max-shifted softmax for numerical stability.
  - Examples:
      - `softmax_type='stable'`: rPCC baseline.
      - `softmax_type='normal'`: sPCC baselines.
- Notes:
  - Unknown values fail through softmax lookup.
  - `stable` is safer for large logits; `normal` matches the static frozen baselines.

### `softmax_k`

- Function: Sets the softmax scale used in update equations.
- Arguments:
  - Type/options: numeric.
  - Description:
      - Larger values sharpen softmax probabilities.
      - Smaller values flatten softmax probabilities.
      - Recurrent models: applies to the classification-error and `Uo` update paths (the
        legacy dynamics `c`) and to reported `Jc`, so the cost tracks the objective whose
        gradient drives learning; evaluation guessing/diagnostics use `softmax_k_eval` instead.
      - Static models: applies everywhere (costs, gradients, guesses).
  - Examples:
      - `softmax_k=1`: rPCC and sPCC baselines.
- Notes:
  - Large values sharpen probabilities and can magnify numerical issues with `softmax_type='normal'`.
  - Changing this in recurrent configs changes learning dynamics (not just diagnostics).

## RPCC ONLY
### INPUTS
### `num_ts`

- Function: Sets recurrent sequence length.
- Arguments:
  - Type/options: integer timestep count for rPCC; `None` placeholder in sPCC configs.
  - Description:
      - Integer: number of timesteps in each recurrent input sequence.
      - `None`: static placeholder only; static code does not use recurrent timesteps.
  - Examples:
      - `num_ts=98`: rPCC CVCV12 baseline.
      - `num_ts=None`: sPCC placeholder.
- Notes:
  - Required by rPCC; omitting it raises `Config missing required keys: num_ts`.
  - Must agree with the recurrent dataset time axis; inputs with NaNs at timestep 0 raise `Input contains NaNs at timestep 0; no valid recurrent timesteps to process.`

### TRAINING / HYPERPARAMETERS
### `kV`

- Function: Sets recurrent-weight learning rates.
- Arguments:
  - Type/options: rPCC dictionary with integer keys `1..num_layers`; no `'o'` key. Static configs use `None`.
  - Description:
      - `kV[i]`: learning rate for recurrent transition weights `V[i]` at layer `i`.
      - `None`: static placeholder only.
  - Examples:
      - `kV={1:0.01, 2:0.01}`: rPCC baseline.
      - `kV=None`: sPCC placeholder.
- Notes:
  - Missing recurrent keys raise `kV missing required key(s): [...]`.
  - An `'o'` key raises `kV must not contain an 'o' key.`

### `ssqr`

- Function: Sets recurrent representation denominators.
- Arguments:
  - Type/options: rPCC dictionary with integer keys `0..num_layers`; no `'o'` key. Static configs use `None`.
  - Description:
      - `ssqr[0]`: input-to-layer-1 representation-error denominator.
      - `ssqr[i]`: inter-layer/top representation-error denominator for recurrent updates.
      - `None`: static placeholder only.
  - Examples:
      - `ssqr={0:10, 1:10, 2:5}`: rPCC baseline.
      - `ssqr=None`: sPCC placeholder.
- Notes:
  - Missing recurrent keys raise `ssqr missing required key(s): [...]`.
  - An `'o'` key raises `ssqr must not contain an 'o' key.`

### `ssqV`

- Function: Sets recurrent transition denominators.
- Arguments:
  - Type/options: rPCC dictionary with integer keys `1..num_layers`; no `'o'` key. Static configs use `None`.
  - Description:
      - `ssqV[i]`: denominator for recurrent transition-weight `V[i]` updates/costs.
      - `None`: static placeholder only.
  - Examples:
      - `ssqV={1:1, 2:1}`: rPCC baseline.
      - `ssqV=None`: sPCC placeholder.
- Notes:
  - Missing recurrent keys raise `ssqV missing required key(s): [...]`.
  - An `'o'` key raises `ssqV must not contain an 'o' key.`

### ARCHITECTURE
### `rc_topdown_cost_denominator`

- Function: Scales recurrent classification context.
- Arguments:
  - Type/options: numeric denominator for rPCC; `None` placeholder in sPCC configs.
  - Description:
      - Numeric value: denominator on the recurrent top-down classification-context correction.
      - `None`: static placeholder only.
  - Examples:
      - `rc_topdown_cost_denominator=2`: rPCC baseline.
      - `rc_topdown_cost_denominator=None`: sPCC placeholder.
- Notes:
  - Required by rPCC; omitting it raises `Config missing required keys: rc_topdown_cost_denominator`.
  - Static configs do not use it.

### `softmax_k_eval`

- Function: Recurrent-only softmax scale for evaluation outputs: classification guessing
  (accuracy) and softmaxed activation diagnostics.
- Arguments:
  - Type/options: positive numeric (recurrent); `None` (static).
  - Examples:
      - `softmax_k_eval=20`: rPCC baseline (legacy `softmax_c=20` scoring).
      - `softmax_k_eval=None`: sPCC baselines.
- Notes:
  - Has no effect on learning or on reported `Jc` (which uses `softmax_k`); it only rescales
    plotted/scored probabilities.
  - Accuracy is argmax-based and therefore invariant to this value.
  - Static models must set `None`; a non-None value raises `softmax_k_eval is recurrent-only...`.
  - Recurrent models require a positive number; otherwise parsing raises `softmax_k_eval must be a positive number for recurrent models.`

## SPCC ONLY
### INPUTS
### `ntiles_per_input`

- Function: Sets static tiling count.
- Arguments:
  - Type/options: `None` or positive integer; rPCC configs use `None`.
  - Description:
      - Positive integer: number of tiles/receptive-field patches per static input.
      - `None`: untiled static input or recurrent placeholder.
  - Examples:
      - `ntiles_per_input=None`: rPCC placeholder or untiled static input.
      - `ntiles_per_input=225`: sPCC natural-image baseline.
      - `ntiles_per_input=16`: sPCC trace baseline.
- Notes:
  - Static models reject booleans, non-integers, and integers `< 1` with `ntiles_per_input must be None or a positive integer for static models.`
  - `'reshaped'` first-to-second connections require this to be set and `flat_input=True`.

### `flat_input`

- Function: Selects static input flattening mode.
- Arguments:
  - Type/options: explicit boolean for sPCC; rPCC configs use `None`.
  - Description:
      - `True`: static input tiles are already flat vectors.
      - `False`: static input retains matrix/spatial dimensions.
      - `None`: recurrent placeholder only.
  - Examples:
      - `flat_input=None`: rPCC placeholder.
      - `flat_input=True`: tiled-flat sPCC baselines.
      - `flat_input=False`: tiled-expanded or untiled-expanded static experiment.
- Notes:
  - Static models require an explicit boolean; otherwise `flat_input must be explicitly set to True or False for static models.`
  - `'reshaped'` first-to-second connections require `flat_input=True`.

### TRAINING / HYPERPARAMETERS
### `ssq`

- Function: Sets static representation denominators.
- Arguments:
  - Type/options: sPCC dictionary with integer keys `0..num_layers`; no `'o'` key. rPCC configs use `None`.
  - Description:
      - `ssq[0]`: input-to-layer-1 representation-error denominator.
      - `ssq[i]`: inter-layer/top representation-error denominator; top key supplies classification pressure in `c1`.
      - `None`: recurrent placeholder only.
  - Examples:
      - `ssq=None`: rPCC placeholder.
      - `ssq={0:1, 1:10, 2:10, 3:10}`: sPCC natural-image baseline.
      - `ssq={0:1, 1:10, 2:10, 3:2}`: sPCC trace baseline.
- Notes:
  - Missing static keys raise `ssq missing required key(s): [...]`.
  - An `'o'` key raises `ssq must not contain an 'o' key.`

### `alph`

- Function: Sets static r-prior coefficients.
- Arguments:
  - Type/options: sPCC dictionary with integer keys `1..num_layers`; no `'o'` key. rPCC configs use `None`.
  - Description:
      - `alph[i]`: coefficient for the static representation prior cost at layer `i`.
      - `None`: recurrent placeholder only.
  - Examples:
      - `alph=None`: rPCC placeholder.
      - `alph={1:1.0, 2:0.05, 3:0.05}`: sPCC baselines.
- Notes:
  - Missing static keys raise `alph missing required key(s): [...]`.
  - An `'o'` key raises `alph must not contain an 'o' key.`

### `lam`

- Function: Sets static U-prior coefficients.
- Arguments:
  - Type/options: sPCC dictionary with integer keys `1..num_layers`; add `'o'` only for static `classif_method='c2'`. rPCC configs use `None`.
  - Description:
      - `lam[i]`: coefficient for the static weight prior cost at layer `i`.
      - `lam['o']`: coefficient for static c2 output-readout prior cost.
      - `None`: recurrent placeholder only.
  - Examples:
      - `lam=None`: rPCC placeholder.
      - `lam={1:0.02, 2:0.00001, 3:0.02}`: sPCC natural-image baseline.
      - `lam={1:0.001, 2:0.001, 3:0.001}`: sPCC trace baseline.
      - `lam={1:0.001, 2:0.001, 3:0.001, 'o':0.001}`: static c2 example.
- Notes:
  - Missing static keys raise `lam missing required key(s): [...]`.
  - `'o'` without `classif_method='c2'` raises `lam may contain an 'o' key only when classif_method='c2'.`
  - Static `classif_method='c2'` without `'o'` raises `lam must include an 'o' key when classif_method='c2'.`

### `plot_receptive_fields`

- Function: Toggles static receptive-field plots.
- Arguments:
  - Type/options: `False`, `True`, `{'mode': None}`, `{'mode': 'all'}`, or `{'mode': {'random_sample': N}}` with positive integer `N`.
  - Description:
      - `False` or `{'mode': None}`: disable receptive-field plots.
      - `True` or `{'mode': 'all'}`: plot receptive fields for all inputs.
      - `{'mode': {'random_sample': N}}`: plot receptive fields for `N` randomly sampled inputs.
  - Examples:
      - `plot_receptive_fields=False`: rPCC placeholder or disabled static plotting.
      - `plot_receptive_fields={'mode': 'all'}`: sPCC natural-image baseline.
      - `plot_receptive_fields={'mode':{'random_sample':5}}`: sPCC trace baseline.
- Notes:
  - Missing/non-bool/non-dict values raise `plot_receptive_fields must be set to a bool or a dict like {'mode': None}, {'mode': 'all'}, or {'mode': {'random_sample': N}}.`
  - Dicts must contain only `mode`; otherwise raises `plot_receptive_fields dict form must contain only the 'mode' key.`
  - Random sample must be an integer `>= 1`; otherwise raises `plot_receptive_fields random_sample must be a positive integer.`
  - Recurrent models reject enabled modes with `plot_receptive_fields is only supported for static models.`
  - Enabled modes require `plot_train=True`; otherwise raises `plot_receptive_fields requires plot_train=True.`
  - Missing tiling metadata can still plot, but the user sees `Warning: tiled receptive-field overlap metadata not found ... Only tiled mosaic reconstructions will be returned.`

## Common Conflict Scenarios

- `plot_softmaxed_activations=True` and `plot_train=False`:
  - Conceptually impossible because activation diagnostics are emitted by the plot-save path.
  - User sees: `plot_softmaxed_activations requires plot_train=True.`
- `plot_softmaxed_activations=True` on sPCC:
  - Conceptually unsupported because current SPCC word datasets lack cohort/rhyme/unrelated lookup maps.
  - User sees: `plot_softmaxed_activations is not supported for static models...`
- Enabled `plot_receptive_fields` and `plot_train=False`:
  - Conceptually impossible because receptive fields are rendered by static plotting.
  - User sees: `plot_receptive_fields requires plot_train=True.`
- Enabled `plot_receptive_fields` on rPCC:
  - Conceptually invalid because recurrent acoustic trajectories have no static receptive-field renderer.
  - User sees: `plot_receptive_fields is only supported for static models.`
- `classif_method='c1'` and `top_lyr_size != num_classes`:
  - Conceptually invalid because c1 treats top-layer activations as class logits directly.
  - User sees: `num_classes must equal top_lyr_size when classif_method='c1'.`
- `classif_method='c2'` without `kU['o']`:
  - Conceptually invalid because the learned output readout cannot update.
  - User sees: `kU must include an 'o' key when classif_method='c2'.`
- Static `classif_method='c2'` without `lam['o']`:
  - Conceptually invalid because static c2 readout prior/regularization lacks a coefficient.
  - User sees: `lam must include an 'o' key when classif_method='c2'.`
- Any `'o'` key in `kr`, `kV`, `alph`, `ssq`, `ssqr`, or `ssqV`:
  - Conceptually invalid because only c2 output readouts use an output key.
  - User sees: `<param> must not contain an 'o' key.`
- `kU['o']` or `lam['o']` while not using `classif_method='c2'`:
  - Conceptually invalid because no output-readout parameter exists.
  - User sees: `<param> may contain an 'o' key only when classif_method='c2'.`
- `architecture='flat_hidden_lyrs'` with `first_to_second_lyr_connection_architecture='reshaped'`:
  - Conceptually invalid because there is no expanded first layer to reshape.
  - User sees: `flat_hidden_lyrs requires first_to_second_lyr_connection_architecture='structured'.`
- `first_to_second_lyr_connection_architecture='reshaped'` without tiled-flat static input:
  - Conceptually invalid because the reshaped `U2` path flattens expanded tiled `r1`.
  - User sees: `first_to_second_lyr_connection_architecture='reshaped' requires ntiles_per_input to be set and flat_input=True.`
- rPCC with static prior costs or denominators set:
  - Conceptually invalid because recurrent update equations currently do not include prior-cost drivers.
  - User sees: `rPCC has no prior cost drivers in its update terms...`
