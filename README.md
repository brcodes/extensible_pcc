# Arbitrarily-Extensible Recurrent or Static Predictive Coding Classifiers (rPCC or sPCC)
## Upgrades hard-coded (i.e., 'inextensible') legacy repo accepted for Rogers et al. 2026 (Neural Computation)
## Extensible Models
### pypredcoding/

- **arbitrarily extensible models** and training, evaluation interface. *Compress, extend, reshape and re-task publication-equivalent models according to user preference.* Often called "-ext" models.
- Check out:
    - `MODEL_MATH.md` (*detailed, comprehensive human-audited equations*)
    - `CONFIG_README.md` (*config API: understand your training entrypoints*)
    - `CONSTRUCTION_HARNESS.md` (*test harness letting you know if you broke something such that arbitrary compression/extension/reshaping/re-tasking cannot take place according to current wiring as of Aug 2026*). 
- Train models by parameterizing with the following format and running:
    ```cd pypredcoding/
    python train_model.py --config config/CONFIG_NAME
    ```

- If `CONFIG_NAME` is:
    ```
    config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt
    config_sPCC_Rogers_2026_raonaturalimages5_frozen_baseline.txt
    config_sPCC_Rogers_2026_trace212_frozen_baseline.txt
    ```
  You will reproduce publication models by classification task (*12 CVCV words*, *5 natural images*, *212 TRACE-like images/words*).
- Regenerate a dataset:
    ```
    python data.py --SETNAME
    ```
- `SETNAME`: 
    - cvcv12, trace212, raonaturalimages5 *(Rogers et al. 2026 datasets)*
    - tom3phon3let_16set, tom3phon3let_4set *(3-letter CVC cochleograms)*
    - Edit `data.py` to make a new set
- Run a grid search:
    ```
    python generate_grid_search_configs.py
    ```
    See docstring in `python generate_grid_search_configs.py`
    ```
    python train_model.py --config config/configs_grid_search_YYMMDD_HHMMSS
    ```


## Inextensible Models
### validated_legacy_models/
- **functionally "inextensible" legacy models** and training, evaluation interface. *Reproduce publication models using the original framework, with original artifacts, plots, and wiring, using a close sibling of the original submitted repository for Rogers et al.* Often called "-inext" models.
    - `cd validated_legacy_models/`
- rPCC
    - `cd recurrentPCC/`
    - `python rPCC_Trainer.py` (12 CVCV)
    - rPCC *all main plots* in ClassAcc...py, LexPred...py, RepSim...py (`see LEGACY_README.md`). See corresponding Rogers et al. 2026 author Kevin Brown for additional plots (e.g. *phonologic clustering*).
- sPCC
    - `cd staticPCC/`
    - `python sPCC_Trainer5Nat.py` (5 Nat Imgs) 
    - `python sPCC_Trainer.py` (212 TRACE-like)
    - sPCC *training accuracy plots* auto-generated (`see LEGACY_README.md`)

## Filetree
```
root/
├── README.md
├── requirements.txt
│
├── pypredcoding/
│ ├── CONFIG_README.md
│ ├── CONSTRUCTION_HARNESS.md
│ ├── TRAINING_KPIs.md
│ ├── MODEL_MATH.md
│ ├── cost_functions.py
│ ├── data.py
│ ├── generate_grid_search_configs.py
│ ├── model.py
│ ├── train_model.py
│ │
│ ├── config/
│ │ ├── config_active.txt
│ │ ├── config_rPCC_Rogers_2026_cvcv12_frozen_baseline.txt
│ │ ├── config_sPCC_Rogers_2026_raonaturalimages5_frozen_baseline.txt
│ │ └── config_sPCC_Rogers_2026_trace212_frozen_baseline.txt
│ │  
│ ├── data/
│ │ ├── Rogers_2026_cvcv12.pydb
│ │ ├── Rogers_2026_raonaturalimages5.pydb
│ │ ├── Rogers_2026_trace212.pydb
│ │ ├── tom_3phon3let_16set_nanpad.pydb
│ │ ├── tom_3phon3let_4set_nanpad.pydb
│ │ ├── metadata/...
│ │ └── raw/...
│ │ 
│ ├── documents/
│ │ ├── rpcc-cochleogram-experiment-docs-2026/...
│ │ ├── tiling-and-rfs-info/...
│ │ └── REPRODUCE_ROGERS_ETAL_2026.md
│ │ 
│ ├── log/... (populate by user)
│ ├── models/... (populate by user)
│ ├── results/... (populate by user)
│ ├── test_results/... (populate by user)
│ │
│ └── tests/
│   ├── parity_support.py
│   ├── test_equivalence_3epoch.py
│   ├── test_smoke_1input.py
│   └── test_model_construction.py
│
└── validated_legacy_models/
    ├── LEGACY_README.md
    ├── recurrentPCC/...
    └── staticPCC/...
```
### Attribution

Portions of this software were developed by Bryce Rogers at Oregon State University
in connection with graduate research, as well as by Monica Yin-Chen Li at the University of Connecticut in connection with graduate research. Additional, non-academic development by B.R. took place after 2025.

This statement describes the project's history and does not identify the
copyright holder, grant reuse rights, or imply endorsement by Oregon State
University or the University of Connecticut.
