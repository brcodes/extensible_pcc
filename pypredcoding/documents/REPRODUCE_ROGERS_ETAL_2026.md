# This repository deterministically reproduces Rogers et al. 2026 results. Two part premise:
## 1. Both *extensible* and *inextensible* frameworks are fully equivalent to each other under the right parameterizations
Extensible and legacy models identically use 2 seeds:
- weight/rep inits `seed_init=1`
- per-epoch input shuffling `seed_shuffle=1`

Seeding and architecture make the them mathematically/architecturally/statistically identical despite ordering of floating-point operations (e.g. batched matmul vs per-module dots). This is *expected given ext's modularity, ok, and unbiased*. This claim has been tested on all three publication classification tasks (cvcv12, trace212, raonaturalimages5).

This is substantiated with *(a)* extremely tight, passing 1-input and 3-epoch parity tests (see below) for all three tasks, *(b)* classification accuracy identity over full run curves after subtracting for minor unbiased and expected floating point operation noise between the two frameworks, for all three tasks (see `LEGACY_REAMDE.md` doc esp. bottom), *(c)* extensive master equations document formulated using only repository code by state of the art Fable 5, line by line human-reviewed by Rogers.

## 2. Both frameworks can deterministically reproduce Rogers et al. 2026 behavior where evaluation domains overlap

Original s/rPCC published runs used only a `seed_init`, with OS for shuffling. Our two-seed method has been shown to replicate those results nonetheless: 

- sPCC 5 Natural Images: 100% classification accuracy by 1000 epochs
- sPCC 212 Trace-like words: ~80% class. accuracy by ~ 300 epochs.
- rPCC 12 CVCV words: 100% classification by ~ 2000 epochs.
- rPCC 12 CVCV words, *-inext only***: main publication evaluation suite in `validated_legacy_models/` -- `see LEGACY_README.md`

***-inext only* is a function of evaluation scope:
- the *pcc-extensible* framework only evaluates: 
    -  training accuracy and costs per epoch
    -  lexical activations (rPCC only)
    -  receptive field plotting (sPCC only)
    -  cost/accuracy related KPIs (`cf. TRAINING_KPIs.md`)
- the *pcc-inextensible* framework covers
    -  many of those *AND* a suite of other rPCC-specific metrics and plots *published in Rogers et al.*
- Therefore rPCC-ext is taken to deterministically reproduce all of the publication/rPCC-inext eval suite given its demonstrated determinism with rPCC-inext across central overlapping domains of evaluation.

## A. Parity tests tell you if you have broken inter-framework determinism (optional)
- Keeping legacy code around is optional. The repo will function with it ablated. But if you do keep it around, parity tests answer:
- Have I *broken something in my extensible codebase or in my inextensible codebase such that the models when commensurately parameterized for Rogers et al. tasks no longer reproduce each other's training trajectories?* 
- Assesses differences down to weight/rep floating point tolerance.
- One-input smoke test (train for 1 input)
```bash
python tests/test_smoke_1input.py \
    --model MODEL_TYPE --task TASK_NAME \
    --seed-init 1 --seed-shuffle 1
```
- Three-epoch equivalence test (train for 3 epochs)
```bash
python tests/test_equivalence_3epoch.py \
   --model MODEL_TYPE --task TASK_NAME \
   --seed-init 1 --seed-shuffle 1
```
- Args
`--model`: rPCC, sPCC
`--task`: cvcv12, trace212, raonaturalimages5

