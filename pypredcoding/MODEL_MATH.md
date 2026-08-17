# MODEL_MATH.md — Master Mathematical Reference for Extensible sPCC / rPCC

This document is the single mathematical reference for the extensible static (sPCC) and
recurrent (rPCC) predictive-coding classifiers implemented in
`pypredcoding/model.py` and `pypredcoding/cost_functions.py`.

All math is fully general: no fixed layer counts, sizes, etc. Every math
symbol is mapped to the run-config parameter that controls it (see `config/CONFIG_README.md`
for parsing/use details of each parameter). Each section opens with a **parameter map**
that binds config keys to the math variables used in that section.

Equations are written exactly as implemented. Where a code path differs from the
generalized path, the difference is stated explicitly.

---

## 1. Global Notation and Conventions

### Parameter map

- $n$ = `num_layers` — number of representation layers above the input.
- $\ell_1, \ldots, \ell_{n-1}$ = `hidden_lyr_sizes` (exactly $n-1$ entries).
- $\ell_n$ = `top_lyr_size` — **the top representation $r_n$ always has size `top_lyr_size`**.
- $C$ = `num_classes` — label width.
- $r_0$ has shape `input_shape`.
- $T$ = `num_ts` (rPCC only) — recurrent sequence length.
- $y$ — one-hot label vector, $y \in \mathbb{R}^{C}$.
- $s(\cdot\,; \kappa)$ — softmax with sharpness $\kappa$ = `softmax_k`, implementation selected by `softmax_type`. rPCC evaluation/guessing/diagnostics use $\kappa_{\text{eval}}$ = `softmax_k_eval` instead (§2.4).

### Layer indexing

Layers are indexed $i = 0, 1, \ldots, n$. Layer $0$ is the input; layers $1..n-1$ are hidden;
layer $n$ is the top layer. `num_layers=n` counts the non-input layers, so a model always has
$r_0$ (input) plus $r_1 \ldots r_n$.

### Error nomenclature (used throughout)

- **$e$ — between-layer (inter-layer) prediction errors.** The difference between a layer's
  state and the generative prediction descending from the layer above:

$$
e_i = r_i - U_{i+1} r_{i+1}
$$

  These exist in both sPCC and rPCC (in rPCC they are time-indexed and computed on
  predicted/"bar" states).

- **$e'$ — within-layer (recurrent/temporal) prediction errors.** rPCC only. The difference
  between a layer's corrected state and its own temporal prediction through the recurrent
  transition weight $V$:

$$
e'_{i,t} = \hat{r}_{i,t} - \bar{r}_{i,t} = \hat{r}_{i,t} - \hat{V}_{i,t-1}\hat{r}_{i,t-1}
$$

  $e'$ drives $V$ learning; it appears in no representation update and no cost term directly,
  but influences both through $V$.

### Hat/bar notation (rPCC)

- $\bar{x}_t$ ("bar") — the *predicted* state at timestep $t$, produced by propagating the
  previous corrected state forward in time.
- $\hat{x}_t$ ("hat") — the *corrected* state at timestep $t$, after error-driven updates.

These correspond to code arrays `rbar`, `rhat`, `Ubar`, `Uhat`, `Vbar`, `Vhat`.

### Symmetric error accounting (the factor of 2)

Every between-layer error $e_i$ ($1 \le i \le n-1$) is counted twice in the representation
cost: once as the top-down error of layer $i$ and once as the bottom-up error of layer
$i+1$. The input error $e_0$ is counted once (nothing sits below the input). This is
intentional and preserved from the validated baselines.

---

## 2. Shared Components

### 2.1 Prior distributions (initialization / reset)

#### Parameter map

- `r_prior_dist` — sampler for representation states $r_i$ (initialization and per-sample reset).
- `U_prior_dist` — sampler for generative weights $U_i$ (and output readout $U^o$).
  **rPCC transition weights $V_i$ also draw from `U_prior_dist`.**
- `r_reset_mode` — per-sample reset policy.
- `seed_init` — NumPy seed applied before fresh construction.

#### Samplers

| `*_prior_dist` value | Draw |
|---|---|
| `('gaussian', mean, scale)` | $x \sim \mathcal{N}(\text{mean}, \text{scale})$ |
| `('sparse_kurtotic', mean, scale)` | $x \sim \mathrm{Laplace}(\text{mean}, \text{scale})$ |
| `('pseudo_sparse_kurtotic',)` | $x = 0$ (deterministic zeros) |
| `('uniform_random', shift)` | $x \sim \mathcal{U}[0,1) + \text{shift}$ |

#### Reset policy

Representations $r_1..r_n$ (and rPCC's context state $r_c$) are reset from `r_prior_dist`
before every sample:

- `r_reset_mode='reset_from_single_sample'` — one draw per distinct layer size is cached at
  training start; every reset loads a **copy** of that cached draw (fast, not a fresh random
  draw each sample).
- `r_reset_mode='reset_with_continual_resampling'` — a fresh draw per layer per sample.

### 2.2 Prior costs (sPCC only)

#### Parameter map

- `r_prior_cost` — selects $g$, the representation prior; coefficient per layer $\alpha_i$ = `alph[i]`.
- `U_prior_cost` — selects $h$, the weight prior; coefficient per layer $\lambda_i$ = `lam[i]`.
- $D_r$ = `r_prior_cost_denominator` — scales $g'$ in representation updates.
- $D_U$ = `U_prior_cost_denominator` — scales $h'$ in weight updates.

rPCC has **no prior-cost drivers** in any update or cost term; all four keys must be `None`
for rPCC (prior *distributions* still control rPCC initialization).

#### Functional forms

Both $g$ (on $r$, coefficient $\alpha$) and $h$ (on $U$, coefficient $\lambda$) use the same
two forms. Sums run over all elements of the array.

Gaussian prior (`'gaussian'`):

$$
g(r;\alpha) = \alpha \sum_j r_j^2,
\qquad
g'(r;\alpha) = 2\alpha r
$$

Sparse-kurtotic prior (`'sparse_kurtotic'`):

$$
g(r;\alpha) = \alpha \sum_j \log\!\left(1 + r_j^2\right),
\qquad
g'(r;\alpha) = \frac{2\alpha r}{1 + r^2} \quad (\text{elementwise})
$$

Identical forms hold for $h(U;\lambda)$ and $h'(U;\lambda)$.

**Denominator placement.** The prior *cost values* $g_i(r_i)$ and $h_i(U_i)$ enter the
representation cost $J_r$ **without** denominators. The denominators $D_r$ and $D_U$ appear
only in the *update* equations, scaling the gradient terms $g'$ and $h'$.

### 2.3 Activation function

#### Parameter map

- `activ_func` — `'linear'` or `'tanh'`; selects $f$ with derivative operator $F$.

The general predictive-coding formulation passes generative predictions through an
activation, $r_{i-1} \approx f(U_i r_i)$, with the derivative $F$ modulating gradients.
The implemented transforms are:

$$
\text{linear:}\quad f(x) = x,\quad F = I
\qquad\qquad
\text{tanh:}\quad f(x) = \tanh(x),\quad F = \mathrm{diag}\!\left(1 - f(x)^2\right)
$$

**Scope of application.** $f$ and $F$ are applied to every generative prediction $U_i r_i$
($i = 1..n$) in both sPCC and rPCC: bottom-up prediction errors $r_{i-1} - f(U_i r_i)$ (with
$F$ modulating the transposed-gradient term in $r$ updates and the error term in $U$
updates), top-down prediction terms $f(U_{i+1} r_{i+1}) - r_i$, representation costs, and
the static receptive-field reconstruction cascade. Two paths are deliberately excluded:
(1) temporal predictions $V_i \hat{r}_{i,t-1}$ are **always linear** — $f$ never touches a
$V$ product (a deliberate deviation from the legacy rPCC, which passed $V$ products through
$f$); (2) classification paths ($s(r_n)$, $s(U^o r)$) take raw logits — softmax is the
nonlinearity there. In code, `self.f` applies the activation and `self.F_mult(f_x, e)`
computes $F e = f'(x) \odot e$ elementwise, which generalizes to the tiled tensor
architectures. With `'linear'` these insertions are exact identities, so the frozen
baselines are unchanged. Equations elsewhere in this document are written for the linear
baseline; for `'tanh'`, insert $f$/$F$ at the sites listed here.

### 2.4 Softmax

#### Parameter map

- `softmax_type` — `'normal'` or `'stable'` (max-subtraction for numerical stability).
- $\kappa$ = `softmax_k` — sharpness used in update equations.
- $\kappa_{\text{eval}}$ = `softmax_k_eval` — rPCC-only sharpness for evaluation/guessing/diagnostics; set `None` in sPCC configs.

$$
s(z;\kappa)_j = \frac{e^{\kappa z_j}}{\sum_m e^{\kappa z_m}}
\qquad\qquad
s_{\text{stable}}(z;\kappa)_j = \frac{e^{\kappa (z_j - \max_m z_m)}}{\sum_m e^{\kappa (z_m - \max_m z_m)}}
$$

**Where $\kappa$ applies:**

- sPCC uses the configured $\kappa$ everywhere (costs, gradients, guesses); `softmax_k_eval` is
  inert (`None`).
- rPCC uses $\kappa$ (`softmax_k`) in the *gradient/error* paths (the classification error
  entering $\hat{r}_c$, and the $U^o$ update) **and** in the reported classification cost
  $J_c$, so the cost tracks the objective whose gradient drives learning
  ($\nabla_z[-y^\top \log s(z;\kappa)] = \kappa\,(s(z;\kappa) - y)$ — temperature enters the
  derivative, so cost and gradient must share $\kappa$). $\kappa_{\text{eval}}$
  (`softmax_k_eval`) applies only to evaluation outputs: classification *guesses* (accuracy
  metrics) and diagnostic activation traces. Baseline configs set `softmax_k=1`,
  `softmax_k_eval=20`, reproducing the legacy split (dynamics $c=1$, scoring `softmax_c=20`).

### 2.5 Classification methods (both families)

#### Parameter map

- `classif_method` — `'c1'`, `'c2'`, or `None`.
- `num_classes` = $C$; `top_lyr_size` = $\ell_n$.
- `kU['o']` = $k_{U,o}$ — output-readout learning rate (c2 only).
- `lam['o']` — required by config validation for **static** c2, but see note below.

| Mode | Top layer semantics | Readout | Constraint |
|---|---|---|---|
| `c1` | $r_n$ *is* the class-logit vector | $p = s(r_n)$ | $\ell_n = C$ required |
| `c2` | $r_n$ is latent; learned readout maps to logits | $p = s(U^o r_n)$, $U^o \in \mathbb{R}^{C \times \ell_n}$ | $\ell_n$ free; `kU['o']` required |
| `None` | unsupervised; no classification pressure | — | no accuracy/Jc metrics |

With `classif_method=None`, between-layer and prior learning still proceed; the top layer's
classification drive is exactly $0$ (the top error-denominator key — `ssq[n]` / `ssqr[n]` —
is still required by config but multiplies zero).

**Note on `lam['o']`.** Static c2 config validation requires `lam['o']`, but no prior term
$h(U^o)$ exists in the current static classification cost or $U^o$ update. The parameter is
currently inert in the math. **TODO:** wire an $h(U^o;\lambda_o)$ prior cost and its
gradient into the static c2 classification cost and $U^o$ update, or drop the config
requirement.

---

## 3. Static PCC (sPCC)

### 3.1 Parameter map

- $n$ = `num_layers`; $\ell_i$ from `hidden_lyr_sizes` and `top_lyr_size`; $C$ = `num_classes`.
- $\sigma_i^2$ = `ssq[i]` for $i = 0..n$ — error denominators. `ssq[n]` scales the top-layer
  classification drive (there is no generative layer above $n$).
- $k_{r,i}$ = `kr[i]` — representation update rates, $i = 1..n$.
- $k_{U,i}$ = `kU[i]` — weight update rates, $i = 1..n$; $k_{U,o}$ = `kU['o']` (c2 only).
- $\alpha_i$ = `alph[i]`, $\lambda_i$ = `lam[i]` — prior coefficients, $i = 1..n$.
- $D_r$ = `r_prior_cost_denominator`, $D_U$ = `U_prior_cost_denominator`.
- `architecture` — `'flat_hidden_lyrs'` or `'expand_first_lyr'`.
- `first_to_second_lyr_connection_architecture` — `'structured'` or `'reshaped'`.
- $P$ = `ntiles_per_input` (tiled static input), `flat_input` — tile flattening flag.
- `update_method` — see §3.8.

### 3.2 Representations and weights: shapes by architecture

The first-layer size is $\ell_1$ (`hidden_lyr_sizes[0]`, or `top_lyr_size` when $n=1$).

#### `architecture='flat_hidden_lyrs'` (requires `first_to_second_lyr_...='structured'`)

Every representation above the input is a flat vector; a single $r_1$ predicts the entire
input at once.

| Object | Shape |
|---|---|
| $r_0$ | `input_shape` (any dimensionality) |
| $r_i$, $i \ge 1$ | $\mathbb{R}^{\ell_i}$ |
| $U_1$ | `input_shape` $\times\ \ell_1$ (prediction $U_1 r_1$ contracts the last axis) |
| $U_i$, $i \ge 2$ | $\mathbb{R}^{\ell_{i-1} \times \ell_i}$ |

#### `architecture='expand_first_lyr'`

The first layer expands over tiles/spatial positions of the input — one $\ell_1$-sized
submodule per tile (receptive-field style). Three input cases:

**Tiled flat** (`ntiles_per_input`$=P$, `flat_input=True`, `input_shape=(P, m)`):

| Object | Shape | Prediction semantics |
|---|---|---|
| $r_0$ | $\mathbb{R}^{P \times m}$ | |
| $r_1$ | $\mathbb{R}^{P \times \ell_1}$ | |
| $U_1$ | $\mathbb{R}^{P \times m \times \ell_1}$ | per-tile: $(U_1 r_1)_p = U_1[p]\, r_1[p]$ |

**Tiled non-flat** (`flat_input=False`, `input_shape=(a, b, h, w)` — a 2-D grid of 2-D tiles):

| Object | Shape | Prediction semantics |
|---|---|---|
| $r_0$ | $\mathbb{R}^{a \times b \times h \times w}$ | |
| $r_1$ | $\mathbb{R}^{a \times b \times \ell_1}$ | |
| $U_1$ | $\mathbb{R}^{a \times b \times h \times w \times \ell_1}$ | per grid position $(i,j)$: $U_1[i,j]\, r_1[i,j]$ |

**Untiled** (`ntiles_per_input=None`): $r_1 \in \mathbb{R}^{\ell_1}$ — degenerates to the flat case.

#### First-to-second connection: `'structured'` vs `'reshaped'`

This controls only $U_2$ (and $U_n$ when $n = 2$):

- **`'structured'`**: $U_2$ preserves the first-layer structure —
  $U_2 \in \mathbb{R}^{\text{shape}(r_1) \times \ell_2}$, prediction contracts the last axis
  (per-tile top-down prediction).
- **`'reshaped'`** (Li legacy; requires tiled-flat input): the expanded $r_1$ is flattened
  before the second-layer connection —

$$
U_2 \in \mathbb{R}^{(P \cdot \ell_1) \times \ell_2},
\qquad
U_2 r_2 \ \text{is reshaped back to}\ \text{shape}(r_1),
\qquad
U_2^\top \mathrm{vec}(e_1) \in \mathbb{R}^{\ell_2}
$$

  The layer-1 transpose path also uses the Li per-tile transpose ($U_1^\top$ transposes the
  last two axes per tile).

#### Layers $\ge 3$ and the readout

$U_i \in \mathbb{R}^{\ell_{i-1} \times \ell_i}$ for $3 \le i \le n$ (ordinary matrices;
all representations above layer 2 are vectors). For c2:
$U^o \in \mathbb{R}^{C \times \ell_n}$.

### 3.3 Between-layer errors

$$
e_0 = r_0 - U_1 r_1
\qquad\qquad
e_i = r_i - U_{i+1} r_{i+1}, \quad 1 \le i \le n-1
$$

Products and outer products respect the architecture semantics of §3.2 (per-tile einsums for
`expand_first_lyr`; flatten/reshape for `'reshaped'`).

### 3.4 Representation cost $J_r$

For general $n \ge 2$:

$$
J_r
= \frac{1}{\sigma_0^2}\lVert e_0 \rVert^2
+ 2\sum_{i=1}^{n-1}\frac{1}{\sigma_i^2}\lVert e_i \rVert^2
+ \sum_{i=1}^{n} g(r_i;\alpha_i)
+ \sum_{i=1}^{n} h(U_i;\lambda_i)
$$

where $\lVert \cdot \rVert^2$ sums squares over all array elements. The factor of 2
implements the symmetric bidirectional accounting of §1. Prior costs carry no denominators
here (see §2.2).

Special case $n = 1$ (no between-layer error above the input):

$$
J_r = \frac{1}{\sigma_0^2}\lVert e_0 \rVert^2 + g(r_1;\alpha_1) + h(U_1;\lambda_1)
$$

$J_r$ is a diagnostic (summed per epoch over all inputs after each input's updates); it is
not differentiated directly — the updates below are its hand-derived gradients.

### 3.5 Classification cost $J_c$

$$
J_c^{c1} = -\,y^\top \log s(r_n;\kappa)
\qquad\qquad
J_c^{c2} = -\,y^\top \log s(U^o r_n;\kappa)
\qquad\qquad
J_c^{\text{None}} = 0
$$

### 3.6 Representation updates

Each layer's update has (up to) three drives: bottom-up error projected up, top-down error
(or classification drive at the top), and the prior gradient.

**Layer 1** ($n \ge 2$):

$$
\Delta r_1 =
\frac{k_{r,1}}{\sigma_0^2} U_1^\top e_0
\;-\; \frac{k_{r,1}}{\sigma_1^2} e_1
\;-\; \frac{k_{r,1}}{D_r} g'(r_1;\alpha_1)
$$

**Middle layers** $2 \le i \le n-1$:

$$
\Delta r_i =
\frac{k_{r,i}}{\sigma_{i-1}^2} U_i^\top e_{i-1}
\;-\; \frac{k_{r,i}}{\sigma_i^2} e_i
\;-\; \frac{k_{r,i}}{D_r} g'(r_i;\alpha_i)
$$

**Top layer** $i = n$ — the top-down error is replaced by the classification drive
$e^{\text{class}}$:

$$
\Delta r_n =
\frac{k_{r,n}}{\sigma_{n-1}^2} U_n^\top e_{n-1}
\;+\; \frac{k_{r,n}}{\sigma_n^2}\, e^{\text{class}}
\;-\; \frac{k_{r,n}}{D_r} g'(r_n;\alpha_n)
$$

with

$$
e^{\text{class}} =
\begin{cases}
y - s(r_n;\kappa) & \text{c1} \\[4pt]
{U^o}^\top\!\left(y - s(U^o r_n;\kappa)\right) & \text{c2} \\[4pt]
0 & \text{None}
\end{cases}
$$

**Special case $n = 1$**: layer 1 *is* the top layer — its update combines the input
bottom-up drive with the classification drive:

$$
\Delta r_1 =
\frac{k_{r,1}}{\sigma_0^2} U_1^\top e_0
\;+\; \frac{k_{r,1}}{\sigma_1^2}\, e^{\text{class}}
\;-\; \frac{k_{r,1}}{D_r} g'(r_1;\alpha_1)
$$

(For $n = 1$ with tiled input and `expand_first_lyr`, `c1`/`c2` are rejected at construction
because the top layer is not a single logit vector.)

### 3.7 Weight updates

Each generative weight learns from the between-layer error below it, outer-multiplied with
the representation above it (outer products follow §3.2 semantics):

$$
\Delta U_1 =
\frac{k_{U,1}}{\sigma_0^2}\, e_0 \otimes r_1
\;-\; \frac{k_{U,1}}{D_U} h'(U_1;\lambda_1)
$$

$$
\Delta U_i =
\frac{k_{U,i}}{\sigma_{i-1}^2}\, e_{i-1} \otimes r_i
\;-\; \frac{k_{U,i}}{D_U} h'(U_i;\lambda_i),
\qquad 2 \le i \le n
$$

**Output readout (c2 only):**

$$
\Delta U^o =
\frac{k_{U,o}}{\sigma_n^2}\left(y - s(U^o r_n;\kappa)\right) r_n^\top
$$

Note the $U^o$ update uses the top error denominator `ssq[n]` and carries **no prior
gradient** (config-required `lam['o']` is currently inert; see the TODO in §2.5).

### 3.8 Update methods

#### Parameter map

- `update_method` — single-key dict: `{'rW_instdeltas_niters': ν}`, `{'r_niters_W': ν}`,
  or `{'r_eq_W': ε}`.

Per input sample, with $r_0$ set to the sample and $r_{1..n}$ reset from the prior:

**`rW_instdeltas_niters` ($\nu$ iterations)** — *simultaneous instantaneous deltas.* Each
iteration computes all $\Delta r_i$ **and** all $\Delta U$ (incl. $\Delta U^o$) from the same
pre-update state, then applies every delta at once. Representations and weights never see
each other's within-iteration changes.

**`r_niters_W` ($\nu$ iterations)** — *relax-then-learn.* The $r$ updates run $\nu$
sequential steps (each step sees the previous step's $r$), then each weight update
($U$, then $U^o$ if c2) runs **once** against the relaxed representations.

**`r_eq_W` ($\varepsilon$ criterion)** — *relax-to-equilibrium-then-learn.* The $r$ updates
iterate until every layer's relative state change falls below the criterion, then each weight
update runs once. The stopping rule, per layer $i$ with initial norm taken at sample onset:

$$
\frac{\lVert r_i^{(k)} - r_i^{(k-1)} \rVert}{\lVert r_i^{(0)} \rVert} \times 100 \;\le\; \varepsilon
\quad \text{for all } i
$$

**The criterion is compared in percent units** (a change of 5% of the initial norm gives a
value of 5). Choose $\varepsilon$ accordingly.

**Evaluation** uses the r-only counterparts of these methods (weights frozen) with the label
zeroed, so no *label information* can inform the inference trajectory. Note that zeroing the
label does **not** zero the classification term: with $L = 0$ the top-down drive becomes
$-\,\mathrm{softmax}(r_n)$, a label-independent suppressive pressure on the top layer that
still shapes the settled state (identical to the legacy `label*0` behavior). Accuracy is then
scored from the settled state (argmax of $p$; ties count as incorrect).

---

## 4. Recurrent PCC (rPCC)

### 4.1 Parameter map

- $n$ = `num_layers`; $\ell_i$ from `hidden_lyr_sizes` and `top_lyr_size`; $C$ = `num_classes`.
- $T$ = `num_ts`; `input_shape` = $(\ell_0,)$ — one acoustic/feature vector per timestep;
  one sample is a matrix $\mathbb{R}^{\ell_0 \times T}$.
- $\sigma_{r,i}^2$ = `ssqr[i]` for $i = 0..n$ — between-layer error denominators; `ssqr[n]`
  scales the top classification pressure.
- $\sigma_{V,i}^2$ = `ssqV[i]` for $i = 1..n$ — within-layer ($e'$) denominators.
- $k_{r,i}$ = `kr[i]`, $k_{U,i}$ = `kU[i]`, $k_{V,i}$ = `kV[i]`; $k_{U,o}$ = `kU['o']` (c2).
- $D_c$ = `rc_topdown_cost_denominator` — scales the classification-context correction.
- `architecture='flat_hidden_lyrs'` is required (no expanded first layer for rPCC).
- Prior costs/denominators must all be `None` (§2.2).
- `update_method` — only `rW_seq_niters` is currently supported for rPCC.

### 4.2 State arrays and shapes

All representations are vectors; all recurrent arrays carry a trailing time axis of length $T$.

| Object | Shape | Meaning |
|---|---|---|
| $r_{0,t}$ | $\mathbb{R}^{\ell_0}$ | input slice at timestep $t$ |
| $\bar{r}_i, \hat{r}_i$ | $\mathbb{R}^{\ell_i \times T}$ | predicted / corrected trajectories, $i = 1..n$ |
| $\bar{r}_c, \hat{r}_c$ | $\mathbb{R}^{\ell_n \times T}$ | classification-context trajectories |
| $\bar{U}_i, \hat{U}_i$ | $\mathbb{R}^{\ell_{i-1} \times \ell_i \times T}$ | time-sliced generative weights |
| $\bar{V}_i, \hat{V}_i$ | $\mathbb{R}^{\ell_i \times \ell_i \times T}$ | time-sliced recurrent transition weights |
| $\bar{U}^o, \hat{U}^o$ | $\mathbb{R}^{C \times \ell_n \times T}$ | time-sliced readout (c2 only) |

The **context state** $r_c$ is a classification-modulated copy of the top layer: it shares
the top layer's size $\ell_n$ and top transition weight $V_n$, receives the classification
correction, and serves as the *driver* for the top generative weight and top transition
learning. With `classif_method=None` it degenerates to $\hat{r}_{c,t} = \hat{r}_{n,t}$.

**Per-sample reset**: $r_{1..n}$ and $r_c$ are redrawn from the `r` prior (§2.1); the
trajectories are zeroed and $\hat{r}_{i,0}$ is seeded with the reset state, denoted
$r_i^{\text{reset}}$ below.

**Valid timesteps**: samples may be NaN-padded on the time axis; only timesteps before the
first NaN are processed ($T_{\text{valid}} \le T$).

### 4.3 Temporal prediction (bars) and within-layer errors $e'$

At each timestep $t$, every layer first predicts itself forward through its transition weight:

$$
\bar{r}_{i,t} = \hat{V}_{i,t-1}\,\hat{r}_{i,t-1}, \quad 1 \le i \le n
\qquad\qquad
\bar{r}_{c,t} = \hat{V}_{n,t-1}\,\hat{r}_{c,t-1}
$$

At $t = 0$ the "previous" corrected state is the per-sample reset state
($\hat{r}_{i,-1} \equiv r_i^{\text{reset}}$, $\hat{r}_{c,-1} \equiv r_c^{\text{reset}}$) and
the $t\!-\!1$ weight slice is slice $0$.

The within-layer error at each layer is then

$$
e'_{i,t} = \hat{r}_{i,t} - \bar{r}_{i,t}
\qquad\qquad
e'_{c,t} = \hat{r}_{c,t} - \bar{r}_{c,t}
$$

i.e., how much the error-driven correction moved the state away from its own temporal
prediction. These drive only the $V$ updates (§4.7).

### 4.4 Between-layer errors

Computed on predicted (bar) states, using the most recent corrected weights
$\hat{U}_{i,t-1}$:

$$
e_{0,t} = r_{0,t} - \hat{U}_{1,t-1}\,\bar{r}_{1,t}
\qquad\qquad
e_{i,t} = \bar{r}_{i,t} - \hat{U}_{i+1,t-1}\,\bar{r}_{i+1,t}, \quad 1 \le i \le n-1
$$

As in sPCC, $e_{i,t}$ acts top-down on layer $i$ and bottom-up on layer $i+1$.

### 4.5 Representation updates (corrections)

**Layer 1** ($n \ge 2$):

$$
\hat{r}_{1,t} = \bar{r}_{1,t}
+ \frac{k_{r,1}}{\sigma_{r,0}^2}\,\hat{U}_{1,t-1}^\top e_{0,t}
- \frac{k_{r,1}}{\sigma_{r,1}^2}\, e_{1,t}
$$

**Middle layers** $2 \le i \le n-1$:

$$
\hat{r}_{i,t} = \bar{r}_{i,t}
+ \frac{k_{r,i}}{\sigma_{r,i-1}^2}\,\hat{U}_{i,t-1}^\top e_{i-1,t}
- \frac{k_{r,i}}{\sigma_{r,i}^2}\, e_{i,t}
$$

**Top layer** (before classification context):

$$
\hat{r}_{n,t} = \bar{r}_{n,t}
+ \frac{k_{r,n}}{\sigma_{r,n-1}^2}\,\hat{U}_{n,t-1}^\top e_{n-1,t}
$$

**Classification context**: the class-space error is computed on the *predicted* top state
$\bar{r}_{n,t}$ with softmax sharpness $\kappa$ = `softmax_k` (baseline configs: 1, the legacy
$c=1$):

$$
e^{\text{class}}_t =
\begin{cases}
s(\bar{r}_{n,t};\kappa) - y & \text{c1} \\[4pt]
\hat{U}^{o\top}_{t-1}\left(s(\hat{U}^o_{t-1}\bar{r}_{n,t};\kappa) - y\right) & \text{c2} \\[4pt]
0 & \text{None}
\end{cases}
$$

$$
\hat{r}_{c,t} = \hat{r}_{n,t}
- \frac{1}{D_c}\cdot\frac{k_{r,n}}{\sigma_{r,n}^2}\, e^{\text{class}}_t
$$

**Special case $n = 1$**: layer 1 is the top layer; its correction uses only the input
drive, and the context correction applies on top of it:

$$
\hat{r}_{1,t} = \bar{r}_{1,t} + \frac{k_{r,1}}{\sigma_{r,0}^2}\,\hat{U}_{1,t-1}^\top e_{0,t},
\qquad
\hat{r}_{c,t} = \hat{r}_{1,t} - \frac{1}{D_c}\cdot\frac{k_{r,1}}{\sigma_{r,1}^2}\, e^{\text{class}}_t
$$

### 4.6 Generative weight updates ($U$, $U^o$)

Bars carry the previous correction forward: $\bar{U}_{i,t} = \hat{U}_{i,t-1}$
(and $\bar{U}^o_t = \hat{U}^o_{t-1}$).

Each update recomputes its prediction error against the layer's **driver** state
$d_{i,t}$ — the corrected state, with the top layer using the classification context:

$$
d_{i,t} =
\begin{cases}
\hat{r}_{i,t} & 1 \le i \le n-1 \\[2pt]
\hat{r}_{c,t} & i = n
\end{cases}
$$

$$
\hat{U}_{1,t} = \bar{U}_{1,t}
+ \frac{k_{U,1}}{\sigma_{r,0}^2}\left(r_{0,t} - \bar{U}_{1,t}\, d_{1,t}\right) d_{1,t}^\top
$$

$$
\hat{U}_{i,t} = \bar{U}_{i,t}
+ \frac{k_{U,i}}{\sigma_{r,i-1}^2}\left(\bar{r}_{i-1,t} - \bar{U}_{i,t}\, d_{i,t}\right) d_{i,t}^\top,
\qquad 2 \le i \le n
$$

**Output readout (c2 only)**, driven by the context state with $\kappa = 1$:

$$
\hat{U}^o_t = \bar{U}^o_t
+ \frac{k_{U,o}}{\sigma_{r,n}^2}\left(y - s(\bar{U}^o_t\,\hat{r}_{c,t};\kappa)\right)\hat{r}_{c,t}^\top
$$

### 4.7 Transition weight updates ($V$) — the $e'$ learners

Bars carry forward as with $U$: $\bar{V}_{i,t} = \hat{V}_{i,t-1}$. The update is a Hebbian
outer product of the within-layer error with the previous corrected state:

$$
\hat{V}_{i,t} = \bar{V}_{i,t}
+ \frac{k_{V,i}}{\sigma_{V,i}^2}\; e'_{i,t}\;\hat{r}_{i,t-1}^\top,
\qquad 1 \le i \le n-1
$$

The top layer uses the context trajectory throughout:

$$
\hat{V}_{n,t} = \bar{V}_{n,t}
+ \frac{k_{V,n}}{\sigma_{V,n}^2}\; e'_{c,t}\;\hat{r}_{c,t-1}^\top
$$

At $t = 0$, $\hat{r}_{i,-1} \equiv r_i^{\text{reset}}$ and
$\hat{r}_{c,-1} \equiv r_c^{\text{reset}}$.

Expanded, this is the "$r - Vr$" form:
$e'_{i,t} = \hat{r}_{i,t} - \hat{V}_{i,t-1}\hat{r}_{i,t-1}$ — the transition weight learns to
predict its own layer's next corrected state.

### 4.8 Per-timestep update order and update method

rPCC currently supports only `update_method={'rW_seq_niters': ν}` (typically
$\nu = 1$). Per valid timestep, each iteration runs in sequence:

1. $\hat{r}$ updates (§4.3–4.5): bars, corrections, context.
2. $\hat{U}$ updates (§4.6).
3. $\hat{U}^o$ update (c2 only).
4. $\hat{V}$ updates (§4.7).

Unlike the static instantaneous-delta method, these run sequentially within a timestep — the
"instantaneous" character comes from the bar/hat time-slicing (each consumer reads the
$t\!-\!1$ slice), not from delta capture.

After each sample, the final timestep's weight slices are broadcast across the whole time
axis, so the next sample's $t=0$ starts from the accumulated weights.

### 4.9 Representation cost $J_r$

Generalized path (all $n$ except the frozen $n=2$ dispatch), summed over valid timesteps
using corrected weights and predicted states:

$$
J_r = \sum_{t=0}^{T_{\text{valid}}-1}\left[
\frac{1}{\sigma_{r,0}^2}\left\lVert r_{0,t} - \hat{U}_{1,t}\,\bar{r}_{1,t} \right\rVert^2
+ 2\sum_{i=1}^{n-1}\frac{1}{\sigma_{r,i}^2}\left\lVert \bar{r}_{i,t} - \hat{U}_{i+1,t}\,\bar{r}_{i+1,t} \right\rVert^2
\right]
$$

No prior costs and no $e'$ terms appear in $J_r$; it accounts only for between-layer errors
(the factor of 2 is the symmetric accounting of §1). $J_r$ is diagnostic only.

**Frozen $n=2$ legacy quirk.** The dedicated 2-layer `rep_cost` (preserved for Li baseline
equivalence) accumulates the layer-2 bottom-up copy *inside* the timestep loop against the
running top-down total, effectively nesting partial sums rather than applying a clean
$2\times$ factor. The generalized path uses the exact expression above. This affects only the
reported $J_r$ magnitude, never learning.

### 4.10 Classification cost, decision rule, evaluation

Costs sum over valid timesteps on the *predicted* state $\bar{r}_n$ (pre-correction) with the
training sharpness $\kappa$ = `softmax_k`, mirroring both the state and the temperature of the
classification-error term whose gradient drives the dynamics:

$$
J_c^{c1} = \sum_{t=0}^{T_{\text{valid}}-1} -\,y^\top \log s(\bar{r}_{n,t};\kappa)
\qquad\qquad
J_c^{c2} = \sum_{t=0}^{T_{\text{valid}}-1} -\,y^\top \log s(\hat{U}^o_t\,\bar{r}_{n,t};\kappa)
$$

Decision at the final valid timestep, scored on the *corrected* state $\hat{r}_n$ (matching
legacy recognition on `softmaxd_r2_hat`; ties count as incorrect):

$$
\hat{y} = \arg\max_j\; p_{T_{\text{valid}}-1,\,j},
\qquad
p_t = \begin{cases} s(\hat{r}_{n,t};\kappa_{\text{eval}}) & \text{c1} \\ s(\hat{U}^o_t\,\hat{r}_{n,t};\kappa_{\text{eval}}) & \text{c2} \end{cases}
$$

**Evaluation** freezes all weights by broadcasting the last timestep slice of every
$\bar{U}/\hat{U}/\bar{V}/\hat{V}$ (and $U^o$) across the time axis, then runs r-only updates
over the sample. Because weights are frozen, the label's influence on the context state
cannot leak into $\bar{r}_n$ or $\hat{r}_n$, so scoring is label-independent.

### 4.11 Frozen $n = 2$ dispatch vs generalized path

`num_layers=2` routes to dedicated 2-layer methods preserved for frozen-baseline
equivalence. Their $r$, $U$, and $V$ update math is exactly the $n=2$ instantiation of
§4.3–4.7; only `rep_cost` differs (§4.9). $n = 1$ and $n \ge 3$ use the generalized methods.

---

## 5. Architectural Special Cases — Worked Instantiations

Concrete instantiations of the general math above. (Sizes are illustrative placeholders,
not baseline references.)

### 5.1 sPCC, $n = 1$, unsupervised (`classif_method=None`)

Config skeleton: `num_layers=1`, `hidden_lyr_sizes=[]`, `top_lyr_size=`$\ell_1$,
`ssq={0:..., 1:...}`, single-key `kr`, `kU`, `alph`, `lam`.

$$
\Delta r_1 = \frac{k_{r,1}}{\sigma_0^2} U_1^\top e_0 - \frac{k_{r,1}}{D_r} g'(r_1;\alpha_1),
\qquad
\Delta U_1 = \frac{k_{U,1}}{\sigma_0^2}\, e_0 \otimes r_1 - \frac{k_{U,1}}{D_U} h'(U_1;\lambda_1)
$$

Pure sparse-coding/reconstruction learning: `ssq[1]` multiplies a zero classification term.
With tiled input + `expand_first_lyr`, this is the *only* legal classification mode for
$n = 1$ (c1/c2 are rejected because the top layer is tiled, not a vector).

### 5.2 sPCC, $n = 2$, `expand_first_lyr` + `'reshaped'` (Li-style shallow classifier)

$r_1 \in \mathbb{R}^{P \times \ell_1}$;
$U_2 \in \mathbb{R}^{(P\cdot\ell_1) \times \ell_2}$ maps the top layer to the flattened
first layer:

$$
e_1 = r_1 - \mathrm{reshape}_{P \times \ell_1}\!\left(U_2 r_2\right),
\qquad
\Delta r_2 \ni \frac{k_{r,2}}{\sigma_1^2}\, U_2^\top \mathrm{vec}(e_1)
$$

The top layer ($=$ layer 2) takes the classification drive through `ssq[2]`.

### 5.3 sPCC, $n \ge 3$, `expand_first_lyr` + `'structured'`

Layer 1 is tiled; $U_2 \in \mathbb{R}^{P \times \ell_1 \times \ell_2}$ predicts *per tile*
($e_1 \in \mathbb{R}^{P \times \ell_1}$; $U_2^\top e_1$ contracts both tile axes to
$\mathbb{R}^{\ell_2}$). Layers $3..n$ are ordinary vector layers with matrix weights. Top
layer takes the classification drive through `ssq[n]`.

### 5.4 sPCC, any $n$, `flat_hidden_lyrs`

All layers vectors; $U_1$ contracts a single $r_1$ against the full input shape. Requires
`'structured'`. This is the direct static analogue of the rPCC layer stack.

### 5.5 sPCC c2 with latent top layer

`top_lyr_size` $\ne$ `num_classes` allowed; add `kU['o']` (and `lam['o']`, config-required).
Top update swaps $y - s(r_n)$ for ${U^o}^\top(y - s(U^o r_n))$; $U^o$ learns via §3.7. The
top representation becomes a latent bottleneck rather than a logit vector.

### 5.6 rPCC, $n = 1$

Single recurrent layer with $V_1$ recurrence and a context state:

$$
\bar{r}_{1,t} = \hat{V}_{1,t-1}\hat{r}_{1,t-1},
\qquad
\hat{r}_{1,t} = \bar{r}_{1,t} + \frac{k_{r,1}}{\sigma_{r,0}^2}\hat{U}_{1,t-1}^\top e_{0,t}
$$

Context, $U_1$ (driver $= \hat{r}_{c,t}$, since layer 1 is the top layer), and $V_1$
(context trajectory) follow §4.5–4.7 with $n = 1$. Requires `ssqr={0:...,1:...}`,
single-key `kr`, `kU`, `kV`, `ssqV`.

### 5.7 rPCC, $n \ge 3$

Middle layers gain the top-down term $-\frac{k_{r,i}}{\sigma_{r,i}^2} e_{i,t}$; only the top
layer owns the context state and classification drive; every layer owns a $V_i$ learning
from its own $e'_{i,t}$.

---

## 6. Quick Cross-Reference: Code Path ↔ Math

| Code (cost_functions.py) | Math |
|---|---|
| `StaticCostFunction.rep_cost_n_1 / _n_2 / _n_gt_eq_3` | §3.4 |
| `StaticCostFunction.r_updates_n_1 / _n_2 / _n_gt_eq_3`, `r_instdeltas` | §3.6 |
| `StaticCostFunction.U_updates_n_1 / _n_gt_eq_2`, `Uo_update` | §3.7 |
| `StaticCostFunction.rn_topdown_upd_c1 / _c2 / _None` | $e^{\text{class}}$, §3.6 |
| `StaticCostFunction.classif_cost_* / classif_guess_*` | §3.5, decision rule |
| `*_fhl / *_e1l / *_e1l_Li` operator families | §3.2 architecture semantics |
| `RecurrentCostFunction.r_updates_n_gteq_1 / _n_2` | §4.3–4.5 |
| `RecurrentCostFunction.U_updates_n_gteq_1 / _n_2`, `Uo_update` | §4.6 |
| `RecurrentCostFunction.V_updates_n_gteq_1 / _n_2` | §4.7 |
| `RecurrentCostFunction.rep_cost_n_gteq_1 / _n_2` | §4.9 |
| `RecurrentCostFunction.recurrent_classif_error_to_r` | $e^{\text{class}}_t$, §4.5 |
| `model.py: update_method_*` | §3.8, §4.8 |
| `model.py: gaussian_prior_costs / sparse_kurtotic_prior_costs` | §2.2 |
| `model.py: linear_activation / tanh_activation / *_error_gradient` | §2.3 |
| `model.py: softmax / stable_softmax` | §2.4 |
| `RecurrentPCC.fill_UsVs_with_last_timestep` | §4.8, §4.10 evaluation freeze |
| `RecurrentPCC.valid_timesteps_until_nan` | $T_{\text{valid}}$, §4.2 |

---

## 7. Constraint Summary

Full validation behavior can be found elsewhere in the repo; the math-relevant constraints are:

- `hidden_lyr_sizes` must have exactly $n - 1$ entries; $r_n$ size $=$ `top_lyr_size` always.
- `c1` requires $\ell_n = C$; `c2` frees $\ell_n$ but requires `kU['o']` (and static `lam['o']`).
- Static: `ssq` keys $0..n$; `kr`, `kU`, `alph`, `lam` keys $1..n$.
- Recurrent: `ssqr` keys $0..n$; `kr`, `kU`, `kV`, `ssqV` keys $1..n$; prior costs/denominators `None`.
- rPCC requires `architecture='flat_hidden_lyrs'` (hence `first_to_second_lyr...='structured'`).
- `'reshaped'` requires `expand_first_lyr` + tiled-flat input (`ntiles_per_input` set, `flat_input=True`).
- Static $n=1$ + tiled + `expand_first_lyr` + `c1`/`c2` is rejected (non-vectorial top layer).
- rPCC supports only `update_method={'rW_seq_niters': ν}`.
- `batch_size=1` (SGD) is the only batch mode.
