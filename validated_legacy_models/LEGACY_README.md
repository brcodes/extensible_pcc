# Legacy ('Inextensible') Recurrent and Static Predictive Coding Classifier (rPCC and sPCC)

## Sibling repo to original submission for Rogers et al. 2026.
- Main findings and figures, original wiring and artifact conventions

Classifying variants of Rao and Ballard (1999)'s predictive coding (PC) model of vision.

rPCC: Kalman-filter-based, pseudo-speech classifier.
sPCC: An image classifier (no over-time input), incl. TRACE-like inputs and natural images.

## Reproduce: Classification Accuracy, Example Input, and Whole-Lexicon Lexical Competition (Figs. 2, 4 and 5 Right Panel)

1. Open `./recurrentPCC/rPCC_Trainer.py`.
2. Set:

   ```python
   num_training_runs = 100
   ```

3. Run the script.
   - This will produce 100 pickles in form: `./recurrentPCC/models_plus_results/rPCC_cvcv12_train/trained_rpcc_N.pkl`, where N = run_index + 1 (i.e., `trained_rpcc_1.pkl` ... `trained_rpcc_100.pkl`).
   - Each is seeded (for both inits and shuffle order) with N = random np seed.
   - Thus `trained_rpcc_1.pkl` uses both seeds = 1, as run idx 0.
   - Checkpointing: None, full runs only, auto-Overwrites all models produced in dir.
4. Open `./recurrentPCC/ClassAcc_ExampInp_LexTimeCourse.py`.
5. Set:

   ```python
   num_training_runs = 100
   print_train_classification_accuracy=True
   print_train_classification_accuracy=True
   print_examp_input=True
   print_lexical_activation_plots=True
   ```
6. Run the script.
   - **Note:**
     - Example Input draws from `./recurrentPCC/data/raw/CVCV_12`
     - Classification Accuracy draws from pickles 1-100
     - Whole-Lexicon Lexical Competition draws only from pickle 1 (error bands are over correct targets, not different models)
   - **Runtime:** About 15-20 minutes per each 5000 ep training run, on a 2019 MacBook Pro. 100 training runs ~ 30 hours.
   - **Faster Eval:** Set num_training_runs = 1 in main(). All downstream eval will be reachable using `trained_rpcc_1.pkl`. Classification Accuracy plots will just show 1-run data in that case.

## Reproduce: Lexical Prediction Paradigm (Fig. 7) and Phonological Clustering Data for Figure 8

1. Make sure you have `./recurrentPCC/models_plus_results/rPCC_cvcv12_train/trained_rpcc_1.pkl` (see above).
2. Open `./recurrentPCC/LexPredParadigm_PhonClusterData.py`.
3. Set:
    ```
    lpp_plot = True
    save_states_words_onsets = True
    ```
5. Run the script.
   - Will produce LPP fig (7)
   - Will produce pickled output necessary for Cluster Decoding (see K.B. for processing script)
   - **Runtime:** Pretty fast.

## Reproduce: Word Report Accuracy w/ Priors, RDM matrices, RSA results (Figs. 10, 12, 13, 14, 15)

1. Make sure you have `./recurrentPCC/models_plus_results/rPCC_cvcv12_train/trained_rpcc_1.pkl`.
2. Open `./recurrentPCC/RepSimAnalysis.py`.
3. Set:

   ```python
   n_sims = 100
   run_sims = True
   show_rdms = True
   show_wra = True
   show_lines = True
   ```

4. Run the script.
   - Will produce RSA calcs (high compute) and figs.
   - Requires np random seed for script main() = 1
   - To reproduce figs again without calculations, set `run_sims = False` (will grab calculation files instead).
   - **Runtime:** About 3 hours per 100 sims calculations, and about 1.33 hrs for first-time RSA figure generation on a 2019 MacBook Pro.

## Reproduce: Static PCC 212 TRACE-like word input classification

1. Open `./staticPCC/sPCC_Trainer.py`.
2. Run the script.
   - Will produce `./staticPCC/models_plus_results/sPCC_trace212_train/` with:
       - `sPCC_train_acc.png` training accuracy plot.
       - `results.pkl` artifact with results.
   - **Runtime:** About 8.33 hours per 1000 epochs, on a 2019 MacBook Pro.
   - **Note:**
       - Generate new parameter sets which sPCC_Trainer consumes by running `sPCC_TrainParams.py`. Produces `sPCC_parameters.pkl`. 
       - Checkpointing, you need set `CLEAR_SAVED_WEIGHTS=False`(grab most recent epoch for resume) and regenerating `sPCC_parameters.pkl`. Set CLEAR...=True to automatically verwrite extant artifacts.

## Reproduce: Static PCC 5 Natural Image classification

1. Open `./staticPCC/sPCC_Trainer5Nat.py`.
2. Run the script.
   - Will produce `./staticPCC/models_plus_results/sPCC_raonaturalimages5_train/`
       - `sPCC_train_acc.png` training accuracy plot.
       - `receptive_fields/` receptive field images.
       - `results.pkl` artifact with results.
   - **Runtime:** About 30 min per 500 epochs, on a 2019 MacMini.
   - **Note:**
       - Generate new parameter sets/checkpointing: same as Trace212 (above), but in the `sPCC_TrainParams5Nat.py`, `sPCC_parameters_5Nat.pkl`, sPCC_Trainer5Nat pathway.

Thank you for using!

# Compare these results with equivalent s/rPCC-extensible runs

## sPCC

### Training accuracy
Compare `sPCC_train_acc.png` (inext) 
to `eval-accuracy-during-training.diagnostics.mod.static.exp_YYMMDD_HHMMSS.EP#.pydb.png` corresponding with your sPCC frozen baseline config runs (ext)

## rPCC

### Training accuracy
Compare `rPCC_class_acc_testing.pdf` (inext; *yes, use the 'testing' pdf as the 'training' pdf does things a bit differently in according with the published results in Rogers et al.*) 
to `eval-accuracy-during-training.diagnostics.mod.recurrent.exp_YYMMDD_HHMMSS.EP#.pydb.png` corresponding with your rPCC frozen baseline config run (ext)

### Lexical activations
Compare `rPCC_lex_act_after_train.pdf` (inext) 
to `avg-activations.diagnostics.mod.recurrent.exp_YYMMDD_HHMMSS.EP#.pydb.png` corresponding with your rPCC frozen baseline config run (ext)