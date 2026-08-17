#!/usr/bin/env python
# coding: utf-8
"""Representational-similarity analysis of a trained recurrent PCC (rPCC).

Load a trained rPCC lexical model, run noisy/primed word-report simulations,
and produce RDM heatmaps, word-report-accuracy bar plots, RSA time-course
lines/bars, and univariate signal-magnitude plots replicating the Blank &
Davis (2016) clarity-by-prior design. Validated legacy reference code.
"""
from __future__ import annotations

# External dependencies
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import sys
from datetime import datetime
import pickle
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path

script_dir = Path(__file__).resolve().parent
repo_root = Path(__file__).resolve().parents[2]
for path in (script_dir, repo_root):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Utils import compute_bootstrap_ci

# pip install git+https://github.com/thelahunginjeet/kbutil.git
# Kevin Brown's mpl plotting wrapper; requires LaTeX
from kbutil.plotting import pylab_pretty_plot as pretty_plot

# Local dependencies
from validated_legacy_models.recurrentPCC.rPCC_Model import *

# Directories that hold shared dataset dumps
RESULTS_DIR = script_dir / "models_plus_results" / "rPCC_cvcv12_train"
PLOT_TIMESTEP_TICKS = [1, 20, 40, 60, 80, 100]
PLOT_TIMESTEP_ONSETS = [2, 21, 40, 59]
PLOT_TIMESTEP_XLIM = (-1, 100)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def latex_safe_text(text: str) -> str:
    """Escape a small set of characters that break with Matplotlib usetex."""
    if not mpl.rcParams.get('text.usetex', False):
        return text
    return (
        text
        .replace('\\', r'\textbackslash{}')
        .replace('{', r'\{')
        .replace('}', r'\}')
        .replace('_', r'\_')
        .replace('%', r'\%')
        .replace('&', r'\&')
        .replace('#', r'\#')
        .replace('^', r'{\char94}')
    )


def determine_resume_stage(stage_paths: list[tuple[str, str]]) -> tuple[str, str | None]:
    """Return the most advanced cached stage that exists on disk."""
    for stage_name, stage_path in stage_paths:
        if os.path.isfile(stage_path):
            return stage_name, stage_path
    return "none", None


def main() -> None:
    """Run the full rPCC representational-similarity analysis pipeline."""
    np.random.seed(1)

    '''
    Preliminary Logic:
    Compute 95% Bootstrap CIs for Blank Davis 2016 Behavioral Word Report Accuracies
    '''



    '''
    Main Logic:
    Load Trained Model
    Run n Number of Word Report Simulations (With 12w Training Lexicon, in Test Mode)
    Average each given signal (eg r1 act, e1 between-layer error, e'2 within-layer error) 
        Over all words and all sims
    Print results for each (of 4) clarity/prior conditions
    '''

    # Load trained model and parameters
    tstart = datetime.now()
    print(tstart)

    trained_model_path = os.path.join(RESULTS_DIR, "trained_rpcc_1.pkl")

    # Neutral, Match
    prime_levels = [0, 0.02]
    # Low Clarity, High Clarity (high noise, low noise)
    noise_levels = [0.55, 0.275]
    errortype = ('ci', 95)

    # Number of simulations 
    n_sim = 100
    run_sims = False

    show_rdms = True
    show_wra = True
    show_lines = True

    print(f"Running new simulations?: {run_sims}")
    print(f"Number of simulations {n_sim}")
    print(f"Noise levels: {noise_levels}, Prime levels: {prime_levels}")
    print(f"Error type: {errortype}")
    print(f"Show RDMs: {show_rdms}, Show WRA: {show_wra}, Show Lines: {show_lines}")
    print(f'\n')

    rsa_results_filename = f'rsa_results_{n_sim}sim_noise{noise_levels}.pkl'
    rsa_results_path = os.path.join(RESULTS_DIR, rsa_results_filename)

    df_2_filename = f'df_2_{n_sim}sim_noise{noise_levels}.pkl'
    df_2_path = os.path.join(RESULTS_DIR, df_2_filename)

    g_all_filename = f"g_all_{n_sim}sim_noise{noise_levels}.pkl"
    g_all_path = os.path.join(RESULTS_DIR, g_all_filename)

    rsa_lines_filename = f"rsa_lines_{n_sim}sim_noise{noise_levels}.pkl"
    rsa_lines_path = os.path.join(RESULTS_DIR, rsa_lines_filename)

    univariate_lines_filename = f"univariate_lines_{n_sim}sim_noise{noise_levels}.pkl"
    univariate_lines_path = os.path.join(RESULTS_DIR, univariate_lines_filename)

    stage_hierarchy = [
        ("rsa_lines", rsa_lines_path),
        ("g_all", g_all_path),
        ("df_2", df_2_path),
        ("rsa_results", rsa_results_path)
    ]

    use_cached = not run_sims

    if run_sims:
        print('run_sims=True → skipping all cache loads and writing fresh artifacts to RESULTS_DIR.')
    else:
        resume_stage, resume_path = determine_resume_stage(stage_hierarchy)
        if resume_stage != "none":  
            print(f"Detected cached stage '{resume_stage}' at {resume_path}. Resuming from there.")
        else:
            print('No cached intermediates detected; will attempt to load raw simulation results if present.')
        print('')

    rsa_lines_cache_available = use_cached and os.path.isfile(rsa_lines_path)
    # df_2 is always required for univariate analysis
    requires_df2 = True
    if requires_df2:
        print('df_2-dependent stages are required (Univariate analysis, WRA, RDM, or RSA recompute).')
    else:
        print('Skipping df_2-dependent stages.')
    print('')


    if show_wra:

        # Load BD2016 Supplementary Info Table
        bd2016_behWRA = pd.read_csv(script_dir / 'data' / 'S1Data_BlankDavis2016_BehavioralWRA.csv')

        # Save each column by clarity/prior condition (column name) as a list
        bd2016_behWRA_lcn = bd2016_behWRA['lcn'].tolist()
        bd2016_behWRA_lcm = bd2016_behWRA['lcm'].tolist()
        bd2016_behWRA_hcn = bd2016_behWRA['hcn'].tolist()
        bd2016_behWRA_hcm = bd2016_behWRA['hcm'].tolist()

        # Compute 95% Bootstrap CIs for each list
        bd2016_behWRA_lcn_ci, lcn_mean = compute_bootstrap_ci(bd2016_behWRA_lcn, return_stat=True)
        bd2016_behWRA_lcm_ci, lcm_mean = compute_bootstrap_ci(bd2016_behWRA_lcm, return_stat=True)
        bd2016_behWRA_hcn_ci, hcn_mean = compute_bootstrap_ci(bd2016_behWRA_hcn, return_stat=True)
        bd2016_behWRA_hcm_ci, hcm_mean = compute_bootstrap_ci(bd2016_behWRA_hcm, return_stat=True)

        # Convert each ci in place to %
        bd2016_behWRA_lcn_ci = [round(x*100, 2) for x in bd2016_behWRA_lcn_ci]
        bd2016_behWRA_lcm_ci = [round(x*100, 2) for x in bd2016_behWRA_lcm_ci]
        bd2016_behWRA_hcn_ci = [round(x*100, 2) for x in bd2016_behWRA_hcn_ci]
        bd2016_behWRA_hcm_ci = [round(x*100, 2) for x in bd2016_behWRA_hcm_ci]

        # Convert each mean in place to %
        lcn_mean = round(lcn_mean*100, 2)
        lcm_mean = round(lcm_mean*100, 2)
        hcn_mean = round(hcn_mean*100, 2)
        hcm_mean = round(hcm_mean*100, 2)

        # Convert each ci to abs diffs from mean
        bd2016_behWRA_lcn_ci_err = [round(abs(x-lcn_mean), 2) for x in bd2016_behWRA_lcn_ci]
        bd2016_behWRA_lcm_ci_err = [round(abs(x-lcm_mean), 2) for x in bd2016_behWRA_lcm_ci]
        bd2016_behWRA_hcn_ci_err = [round(abs(x-hcn_mean), 2) for x in bd2016_behWRA_hcn_ci]
        bd2016_behWRA_hcm_ci_err = [round(abs(x-hcm_mean), 2) for x in bd2016_behWRA_hcm_ci]

        # Format per-condition summary strings
        bd2016_behWRA_lcn_ci_str = 'Low Clarity Neutral: ' + str(bd2016_behWRA_lcn_ci) + ' Mean: ' + str(lcn_mean) + ' Error: ' + str(bd2016_behWRA_lcn_ci_err)
        bd2016_behWRA_lcm_ci_str = 'Low Clarity Match: ' + str(bd2016_behWRA_lcm_ci) + ' Mean: ' + str(lcm_mean) + ' Error: ' + str(bd2016_behWRA_lcm_ci_err)
        bd2016_behWRA_hcn_ci_str = 'High Clarity Neutral: ' + str(bd2016_behWRA_hcn_ci) + ' Mean: ' + str(hcn_mean) + ' Error: ' + str(bd2016_behWRA_hcn_ci_err)
        bd2016_behWRA_hcm_ci_str = 'High Clarity Match: ' + str(bd2016_behWRA_hcm_ci) + ' Mean: ' + str(hcm_mean) + ' Error: ' + str(bd2016_behWRA_hcm_ci_err)

        # Concatenate all strings with newlines
        bd2016emp_txt = 'BD2016 Behavioral Word Report Accuracies CIs\n' + bd2016_behWRA_lcn_ci_str + '\n' + bd2016_behWRA_lcm_ci_str + '\n' + bd2016_behWRA_hcn_ci_str + '\n' + bd2016_behWRA_hcm_ci_str
        print(bd2016emp_txt)


        # Note the strange middle bars' switching, due to ax.errorbar plotting in a different order than sns.barplot
        lower_errors = [bd2016_behWRA_lcn_ci_err[0], bd2016_behWRA_hcn_ci_err[0], bd2016_behWRA_lcm_ci_err[0], bd2016_behWRA_hcm_ci_err[0]]
        upper_errors = [bd2016_behWRA_lcn_ci_err[1], bd2016_behWRA_hcn_ci_err[1], bd2016_behWRA_lcm_ci_err[1], bd2016_behWRA_hcm_ci_err[1]]


    #--------------------------------------------------------------
    # Load trained model artifacts and rebuild the model
    # -------------------------------------------------------------

    results_dict = pickle.load(open(trained_model_path, "rb"))
    if "param" not in results_dict:
        raise KeyError(f"No 'param' entry found in {trained_model_path}; cannot configure simulation parameters.")

    param = results_dict["param"]
    required_keys = ["I_dict", "output_test_df", "output_train_df", "all_weights_df"]
    missing = [key for key in required_keys if key not in results_dict]
    if missing:
        raise KeyError(f"Missing keys {missing} in {trained_model_path}.")

    I_dict = results_dict["I_dict"]
    output_test_df = results_dict["output_test_df"]
    output_train_df = results_dict["output_train_df"]
    all_weights_df = results_dict["all_weights_df"]

    act_func = param['act_func']
    alpha_1 = param['alpha_1']
    alpha_2 = param['alpha_2']
    beta_1 = param['beta_1']
    beta_2 = param['beta_2']
    cohort_len = param['cohort_len']
    epoch_n = param['epoch_n']
    gamma_1 = param['gamma_1']
    gamma_2 = param['gamma_2']
    r1_size = param['r1_size']
    recog_mode = param['recog_mode']
    recog_value = param['recog_value']
    rhyme_len = param['rhyme_len']
    s10 = param['s10']
    s11 = param['s11']
    s21 = param['s21']
    s22 = param['s22']
    s32 = param['s32']
    save_interval = param['save_interval']
    softmax_c = param['softmax_c']
    weight_init_seed = param['weight_init_seed']

    print(f'params are:\n{param}')

    lexicon = list(I_dict.keys())
    print(f'lexicon is:\n{lexicon}')

    epoch_max = max(range(epoch_n))
    save_epochs = [x for x in range(epoch_n) if x == 0 or x % save_interval == save_interval-1]

    I_size = I_dict[lexicon[0]].shape[0]
    L_size = len(I_dict)
    r2_size = L_size

    model = Model(I_size, r1_size, r2_size, L_size, seed=weight_init_seed)

    model.s10 = s10
    model.s11 = s11
    model.s21 = s21
    model.s22 = s22
    model.s32 = s32

    model.alpha_1 = alpha_1
    model.alpha_2 = alpha_2
    model.beta_1 = beta_1
    model.beta_2 = beta_2
    model.gamma_1 = gamma_1
    model.gamma_2 = gamma_2

    model.U1 = all_weights_df.U1.tail(1).squeeze()
    model.U2 = all_weights_df.U2.tail(1).squeeze()
    model.V1 = all_weights_df.V1.tail(1).squeeze()
    model.V2 = all_weights_df.V2.tail(1).squeeze()

    if show_rdms:
        print("Generating HRDM and IRDM plots...")
        pretty_plot(lines=1.5, width=2)
        rdm_cmap = sns.diverging_palette(260, 10, s=75, l=50, as_cmap=True)

        # -----------------------------
        # HRDM (hypothesized lexical geometry)
        # -----------------------------
        h_rdm = {
            "Cohort": np.empty((len(lexicon), len(lexicon))),
            "Rhyme": np.empty((len(lexicon), len(lexicon))),
        }

        for key in h_rdm.keys():
            for i, x in enumerate(lexicon):
                for j, y in enumerate(lexicon):
                    if key == "Cohort":
                        h_rdm[key][i, j] = np.nan if x == y else (0 if x[:2] == y[:2] else 1)
                    elif key == "Rhyme":
                        h_rdm[key][i, j] = np.nan if x == y else (0 if x[-2:] == y[-2:] else 1)

        h_rdm["Both"] = h_rdm["Cohort"] * h_rdm["Rhyme"]

        nrows = 1
        ncols = len(h_rdm)
        subplot_xy = (5, 5)
        figsize = (ncols * subplot_xy[0], nrows * subplot_xy[1])

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        for idx, ax in zip(h_rdm.keys(), np.atleast_1d(axes).flatten()):
            safe_lexicon = [latex_safe_text(word) for word in lexicon]
            hm = sns.heatmap(
                h_rdm[idx],
                cmap=rdm_cmap,
                cbar_kws={"shrink": 0.6},
                mask=np.triu(np.ones_like(h_rdm[idx], dtype=bool)),
                square=True,
                linewidths=0.5,
                xticklabels=safe_lexicon,
                yticklabels=safe_lexicon,
                ax=ax,
            )
            ax.set_title(idx, fontsize=18)
            ax.tick_params(axis='both', labelsize=14)
            ax.tick_params(axis='x', labelrotation=90)
            ax.tick_params(axis='y', labelrotation=0)
            hm.collections[0].colorbar.ax.tick_params(labelsize=14)

        fig.suptitle("HRDM")
        plt.tight_layout()
        plt.show()

        # Standalone HRDM (Both) plot
        hyp_title = "Hypothesized RDM"
        fig, ax = plt.subplots(figsize=(5, 5))
        safe_lexicon = [latex_safe_text(word) for word in lexicon]
        sns.heatmap(
            h_rdm["Both"],
            cmap=rdm_cmap,
            cbar=False,
            mask=np.triu(np.ones_like(h_rdm["Both"], dtype=bool)),
            square=True,
            linewidths=0.5,
            xticklabels=safe_lexicon,
            yticklabels=safe_lexicon,
            ax=ax,
        )
        ax.tick_params(axis='both', labelsize=14)
        ax.tick_params(axis='x', labelrotation=90)
        ax.tick_params(axis='y', labelrotation=0)
        low_color = rdm_cmap(0.0)
        high_color = rdm_cmap(1.0)
        legend_handles = [
            mpl.patches.Patch(facecolor=high_color, edgecolor="black", label="1"),
            mpl.patches.Patch(facecolor=low_color, edgecolor="black", label="0"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
            frameon=False,
            fontsize=13.65,
            handlelength=0.945,
            handleheight=0.945,
        )
        ax.set_title(hyp_title, fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        plt.tight_layout()
        hyp_pdf_title = f"{hyp_title} Rule Based Scale"
        fig.savefig(os.path.join(RESULTS_DIR, f"{hyp_pdf_title}.pdf"), dpi=600)
        plt.show()

        # -----------------------------
        # IRDM (input feature geometry)
        # -----------------------------
        model_inputs_key = range(98)
        model_inputs_rdm = np.full((len(lexicon), len(lexicon)), np.nan)

        for i, targ in enumerate(lexicon):
            for j, nontarg in enumerate(lexicon):
                if i == j:
                    continue

                # Match RSA construction style: dissimilarity at each timestep,
                # then average those dissimilarities across the selected timeframe.
                timestep_dissim = []
                for ts in model_inputs_key:
                    dissim = 1 - pearsonr(I_dict[targ][:, ts], I_dict[nontarg][:, ts])[0]
                    timestep_dissim.append(dissim)

                model_inputs_rdm[i, j] = np.nanmean(timestep_dissim)

        # Standalone IRDM plot (timeframe range(0, 98))
        model_inputs_title = "Model Inputs RDM"
        fig, ax = plt.subplots(figsize=(5, 5))
        safe_lexicon = [latex_safe_text(word) for word in lexicon]
        hm = sns.heatmap(
            model_inputs_rdm,
            cmap=rdm_cmap,
            cbar_kws={"shrink": 0.6},
            mask=np.triu(np.ones_like(model_inputs_rdm, dtype=bool)),
            square=True,
            linewidths=0.5,
            xticklabels=safe_lexicon,
            yticklabels=safe_lexicon,
            ax=ax,
        )
        ax.set_title(model_inputs_title, fontsize=18)
        ax.tick_params(axis='both', labelsize=14)
        irdm_cbar = hm.collections[0].colorbar
        irdm_cbar.ax.tick_params(labelsize=14)
        irdm_cbar.locator = mpl.ticker.LinearLocator(5)
        irdm_cbar.formatter = mpl.ticker.FuncFormatter(lambda val, pos: rf"${val:.1f}$")
        irdm_cbar.update_ticks()
        plt.tight_layout()
        model_inputs_pdf_title = f"{model_inputs_title} Pearson Dissimilarity"
        fig.savefig(os.path.join(RESULTS_DIR, f"{model_inputs_pdf_title}.pdf"), dpi=600)
        plt.show()


    # Test model RSA output with different levels of noise and with priming

    all_test_df = None

    if run_sims is True:

        all_test_df = pd.DataFrame()
        for s in range(n_sim):
            for n in noise_levels:
                for p in prime_levels:
                    for i, k in enumerate(I_dict.keys()):
                        # Set input
                        I = I_dict[k]
                        # Noise settings
                        ## Optional: new seed for each sim
                        # np.random.seed(s)
                        noise = np.random.normal(loc=0, scale=n, size=I.shape)
                        I = I + noise
                        I = np.clip(I, a_min=0, a_max=1)
                        # Prime settings
                        r2_p = [x*p for x in np.eye(L_size)[i]]
                        # Test model, store output
                        df = model.apply_input(k, I, np.zeros(L_size), training=False, af=act_func, r2_priming=r2_p)
                        df["noise"] = n
                        df["prime"] = p
                        df["simulation"] = s
                        # Append single test word output df (df) to all test words output df (all test df), row-wise
                        all_test_df = all_test_df.append(df, ignore_index=True)

        all_test_df.to_pickle(rsa_results_path)
        print(f'Saved {rsa_results_path}')
    else:
        if os.path.isfile(rsa_results_path):
            all_test_df = pd.read_pickle(rsa_results_path)
            print(f"Loaded cached rsa_results from {rsa_results_path}")
        else:
            print(f"No rsa_results cache found at {rsa_results_path}; relying on downstream pickles.")

    df_2 = None
    if requires_df2:
        if use_cached and os.path.isfile(df_2_path):
            print(f"The file {df_2_path} exists. Loading it.")
            df_2 = pd.read_pickle(df_2_path)
            noise_order = df_2.noise.sort_values(ascending=False).unique()
        # Note: this elif duplicates the branch above and is unreachable (kept as-is).
        elif use_cached and os.path.isfile(df_2_path):
            print(f"The file {df_2_path} exists. Loading it.")
            df_2 = pd.read_pickle(df_2_path)
            noise_order = df_2.noise.sort_values(ascending=False).unique()
        else:
            if all_test_df is None:
                raise FileNotFoundError(f"Missing {df_2_path} and no rsa_results cache available to regenerate it.")
            print(f"The file {df_2_path} does not exist. Running calculations now.")
            tstartdf2 = datetime.now()
            ### Calculations for plotting
            all_test_df["softmaxd_r2_hat"] = all_test_df.r2_hat.apply(lambda x: softmax(x, c=softmax_c))

            all_test_df["e10"] = all_test_df.apply(lambda x: x.I - (model.U1 @ x.r1_hat), axis=1)
            all_test_df["e11"] = all_test_df.apply(lambda x: x.r1_hat - x.r1_bar, axis=1)
            all_test_df["e21"] = all_test_df.apply(lambda x: x.r1_hat - (model.U2 @ x.r2_hat), axis=1)
            all_test_df["e22"] = all_test_df.apply(lambda x: x.r2_hat - x.r2_bar, axis=1)

            all_test_df["e10_abs"] = all_test_df.e10.apply(lambda x: np.abs(x))
            all_test_df["e11_abs"] = all_test_df.e11.apply(lambda x: np.abs(x))
            all_test_df["e21_abs"] = all_test_df.e21.apply(lambda x: np.abs(x))
            all_test_df["e22_abs"] = all_test_df.e22.apply(lambda x: np.abs(x))

     

            all_test_df["r1_re"] = all_test_df.apply(lambda x: f(model.V1 @ x.r1_hat, af=act_func) - x.r1_hat, axis=1)
            all_test_df["r2_re"] = all_test_df.apply(lambda x: f(model.V2 @ x.r2_hat, af=act_func) - x.r2_hat, axis=1)
            all_test_df["r1_bu"] = all_test_df.apply(lambda x: model.U1.T @ f_prime(model.U1 @ x.r1_bar, af=act_func) @ (x.I - f(model.U1 @ x.r1_bar, af=act_func)), axis=1)
            all_test_df["r2_bu"] = all_test_df.apply(lambda x: model.U2.T @ f_prime(model.U2 @ x.r2_bar, af=act_func) @ (x.r1_bar - f(model.U2 @ x.r2_bar, af=act_func)), axis=1)
            # all_test_df["r0_td"] = all_test_df.apply(lambda x: f(model.U1 @ x.r1_bar, af=act_func), axis=1)
            # all_test_df["r1_td"] = all_test_df.apply(lambda x: f(model.U2 @ x.r2_bar, af=act_func), axis=1)
            all_test_df["r1_td"] = all_test_df.apply(lambda x: x.r1_bar - f(model.U2 @ x.r2_bar, af=act_func), axis=1)
            
            all_test_df["r1_re_abs"] = all_test_df.r1_re.apply(np.abs)
            all_test_df["r2_re_abs"] = all_test_df.r2_re.apply(np.abs)
            all_test_df["r1_bu_abs"] = all_test_df.r1_bu.apply(np.abs)
            all_test_df["r2_bu_abs"] = all_test_df.r2_bu.apply(np.abs)
            # all_test_df["r0_td_abs"] = all_test_df.r0_td.apply(np.abs)
            all_test_df["r1_td_abs"] = all_test_df.r1_td.apply(np.abs)

            df_2 = all_test_df.copy()
            noise_order = df_2.noise.sort_values(ascending=False).unique()
            df_2.prime = df_2.prime.astype("category")

            feature_list = ['alveolar', 'anterior', 'back', 'central', 'consonantal',
                            'continuant', 'coronal', 'high', 'labial', 'low',
                            'mid', 'nasal', 'palatal', 'reduced', 'round',
                            'sibilant', 'sonorant', 'syllabic', 'tense', 'velar', 'voiced']
            cmap = "bone_r"

            tf_tag = 'Time Steps 1-98'
            ci_tag = "('ci',95)"

            recog_mode = 1
            recog_value = 0
            test_recog_df = (df_2.groupby(["label", "prime", "noise"]).
                                softmaxd_r2_hat.apply(lambda x: recog(x, mode=recog_mode, value=recog_value)).rename("recog_node").to_frame())
            df_2 = pd.merge(df_2, test_recog_df,
                            left_on=["label", "prime", "noise"], right_on=["label", "prime", "noise"],
                            how="left", suffixes=("_old", ""))
            df_2["label_node"] = df_2.apply(lambda x: lexicon.index(x.label), axis=1)
            df_2["accuracy"] = df_2.apply(lambda x: int(x.label_node == x.recog_node), axis=1)
            df_2["accuracy_percent"] = df_2.accuracy * 100

            # Pickle df_2
            with open(df_2_path, "wb") as file_handle:
                pickle.dump(df_2, file_handle)
                print(f'Saved {df_2_path}')

            tenddf2 = datetime.now()
            print('tot time df2 prep', tenddf2-tstartdf2)
    else:
        print('df_2 not required; skipping load/generation.')

    if df_2 is not None:
        # Make a new df called df_2_emp with only columns 'noise' 'prime' 'accuracy_percent'
        # Fill with appropriate B&D (2016) empirical word report values
        df_2_emp = pd.DataFrame({
            'noise': [0.55, 0.55, 0.275, 0.275],
            'prime': [0, 0.02, 0, 0.02],
            'accuracy_percent': [43.45, 79.17, 83.53, 89.68]
        })

        # Text storage
        text_content = ''

    if df_2 is not None and show_rdms:
        print("Generating Empirical RDM for HCM condition (sim 0, all time steps)...")

        emp_component = "r2_hat"
        emp_noise = 0.275
        emp_prime = 0.02
        emp_sim = 0

        df_emp = df_2[(df_2["noise"] == emp_noise) &
                      (df_2["prime"] == emp_prime) &
                      (df_2["simulation"] == emp_sim)]

        if df_emp.empty:
            print(f"No rows found for empirical RDM condition noise={emp_noise}, prime={emp_prime}, simulation={emp_sim}; skipping empirical RDM plot.")
        else:
            df_emp = df_emp.set_index(["label", "timestep"])
            timesteps = sorted(df_2["timestep"].unique())

            # Hypothesized RDM (Both): 0 when either cohort or rhyme overlaps, else 1.
            hrdm_both = np.ones((len(lexicon), len(lexicon)))
            for i, targ in enumerate(lexicon):
                for j, nontarg in enumerate(lexicon):
                    if targ == nontarg:
                        hrdm_both[i, j] = np.nan
                    elif targ[:2] == nontarg[:2] or targ[-2:] == nontarg[-2:]:
                        hrdm_both[i, j] = 0

            # Build empirical RDM at each timestep, then average across all timesteps.
            rdm_by_ts = {}
            for ts in timesteps:
                ts_rdm = np.full((len(lexicon), len(lexicon)), np.nan)
                for i, targ in enumerate(lexicon):
                    for j, nontarg in enumerate(lexicon):
                        if i == j:
                            continue
                        targ_vec = df_emp.loc[(targ, ts), emp_component]
                        nontarg_vec = df_emp.loc[(nontarg, ts), emp_component]
                        corr = pearsonr(targ_vec, nontarg_vec)[0]
                        ts_rdm[i, j] = np.nan if np.isnan(corr) else 1 - corr
                rdm_by_ts[ts] = ts_rdm

            empirical_rdm = np.nanmean(np.stack([rdm_by_ts[ts] for ts in timesteps], axis=0), axis=0)

            # Compute timestep-wise RSA(empirical, hypothesized) and average over timesteps.
            upper_mask = np.triu(np.ones_like(hrdm_both, dtype=bool))
            hrdm_vec = np.ma.masked_array(hrdm_both, mask=upper_mask).compressed()
            rsa_by_timestep = []
            for ts in timesteps:
                emp_vec = np.ma.masked_array(rdm_by_ts[ts], mask=upper_mask).compressed()
                rsa_ts = spearmanr(emp_vec, hrdm_vec)[0]
                rsa_by_timestep.append(rsa_ts)

            avg_rsa = np.nanmean(rsa_by_timestep)

            print(f"Empirical RDM component: {emp_component}")
            print(f"Condition: noise={emp_noise}, prime={emp_prime}, simulation={emp_sim}")
            print(f"Average RSA(empirical, hypothetical) across all timesteps: {avg_rsa:.6f}")
            print("Empirical RDM (rounded to 3 decimals):")
            print(pd.DataFrame(empirical_rdm, index=lexicon, columns=lexicon).round(3))

            emp_title = "Empirical RDM (All Time Steps, HCM Condition)"
            fig, ax = plt.subplots(figsize=(5, 5))
            safe_lexicon = [latex_safe_text(word) for word in lexicon]
            hm = sns.heatmap(
                empirical_rdm,
                cmap=sns.diverging_palette(260, 10, s=75, l=50, as_cmap=True),
                cbar_kws={"shrink": 0.6},
                mask=np.triu(np.ones_like(empirical_rdm, dtype=bool)),
                square=True,
                linewidths=0.5,
                xticklabels=safe_lexicon,
                yticklabels=safe_lexicon,
                ax=ax,
            )
            ax.set_title(emp_title, fontsize=18)
            ax.tick_params(axis='both', labelsize=14)
            emp_cbar = hm.collections[0].colorbar
            emp_cbar.ax.tick_params(labelsize=14)
            emp_cbar.locator = mpl.ticker.LinearLocator(5)
            emp_cbar.formatter = mpl.ticker.FuncFormatter(lambda val, pos: rf"${val:.1f}$")
            emp_cbar.update_ticks()
            plt.tight_layout()
            emp_pdf_title = f"{emp_title} Pearson Dissimilarity"
            fig.savefig(
                os.path.join(RESULTS_DIR, f"{emp_pdf_title}.pdf"),
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.05,
            )
            plt.show()

    if df_2 is not None and show_wra:
        '''
        Combined Word Report Accuracy Fig
        B&D (2016) Replication (L)
        rPCC Simulation (R)
        '''

        plt.style.use('seaborn-paper')
        plt.rcParams['image.aspect'] = 'auto'
        plt.rcParams['figure.dpi'] = 600
        sns.set_palette("colorblind")
        pretty_plot(labelsize=11, fontsize=11, lfontsize=8, width=2)

        # Create a figure with two subplots, side by side
        fig, axs = plt.subplots(1, 2, figsize=(8, 3))
        color_dict = {0: sns.color_palette()[0], 0.02: sns.color_palette()[1]}
        # Assuming df_2_emp is a dataframe with the same structure as df_2
        # Plot the first barplot on the right subplot
        sns.barplot(data=df_2, x="noise", y="accuracy_percent", hue="prime", order=noise_order, errorbar=errortype, ax=axs[1], palette=color_dict)

        # Set the labels and legend for the first subplot
        axs[1].set_title("(b) rPCC", fontsize=15, pad=9)
        axs[1].set_xlabel("")
        axs[1].set_ylabel("")
        axs[1].set_xticklabels(['Low Clarity', 'High Clarity'], fontsize=14)
        prime_levels_as_words = ['Neutral', 'Match']
        patches = [mpl.patches.Patch(color=sns.color_palette()[i], label=t) for i,t in enumerate(prime_levels_as_words)]
        axs[1].get_legend().remove()

        # Grab bar heights for printing
        neut_heights = [round(bar.get_height(),3) for bar in axs[1].containers[0]]
        match_heights = [round(bar.get_height(),3) for bar in axs[1].containers[1]]
        bar_heights = list(zip(neut_heights, match_heights))
        text_content += bd2016emp_txt + '\n\n'
        text_content += 'rPCC word report accuracies\n\n'
        print('rPCC word report accuracies')

        for c, clarity in enumerate(['Low Clarity', 'High Clarity']):
            for p, prior in enumerate(['Neutral', 'Match']):
                bar_height = f'Acc | {clarity} {prior}: {bar_heights[c][p]}'
                print(bar_height)
                text_content += bar_height + '\n'

        text_content += '\nrPCC word report accuracies with 95% bootstrap CIs\n\n'
        for c, clarity in enumerate(['Low Clarity', 'High Clarity']):
            noise_val = noise_order[c]
            for p, prior in enumerate(['Neutral', 'Match']):
                prime_val = prime_levels[p]
                acc_values = df_2.query("noise == @noise_val and prime == @prime_val").accuracy_percent.to_numpy()
                cis_for_condition, mean_for_condition = compute_bootstrap_ci(
                    acc_values,
                    stat='mean',
                    nboot=10000,
                    ci=0.95,
                    return_stat=True,
                )
                err_for_condition = [
                    abs(cis_for_condition[0] - mean_for_condition),
                    abs(cis_for_condition[1] - mean_for_condition),
                ]
                cond_line = (
                    f"Acc+Err | {clarity} {prior}: mean={mean_for_condition:.3f} "
                    f"ci=[{cis_for_condition[0]:.3f}, {cis_for_condition[1]:.3f}] "
                    f"err=[{err_for_condition[0]:.3f}, {err_for_condition[1]:.3f}]"
                )
                print(cond_line)
                text_content += cond_line + '\n'

        # Assuming prime_levels is a list of prime levels for each bar in the left subplot
        color_dict = {0: sns.color_palette()[0], 0.02: sns.color_palette()[1]}
        colors_left = [color_dict[prime] for prime in prime_levels]

        # Plot the second barplot on the left subplot
        barplot = sns.barplot(data=df_2_emp, x="noise", y="accuracy_percent", hue="prime", order=noise_order, errorbar=errortype, ax=axs[0], palette=color_dict)
        x_values = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
        # # B&D Code Values for Emp Accuracies (not used)
        # y_values = [43.45, 83.53, 79.17, 89.68]
        # B&D Supp Info Values for Emp Accuracies, careful to switch the middle two accuracies, standard errors, around
        y_values = [43.45, 83.53, 79.17, 89.68]

        # Lower_errors and upper_errors are your arrays of errors
        errors_95ci = [lower_errors, upper_errors]
        axs[0].errorbar(x_values, y_values, yerr=errors_95ci, fmt='none', color='#444444', capsize=0, elinewidth=3.5, alpha=1)

        # Set the labels and legend for the second subplot
        axs[0].set_title("(a) Empirical", fontsize=15, pad=9)
        axs[0].set_ylabel(r"Word report accuracy (\%)", fontsize=14)
        axs[0].set_xticklabels(['Low Clarity', 'High Clarity'], fontsize=14)
        axs[0].set_xlabel("")
        axs[0].legend(handles=patches, loc="upper left", fontsize="small", title_fontsize="x-small")
        # set y-axis limit
        axs[0].set_ylim([0, 105])

        # Adjust the layout
        plt.tight_layout()
        behavior_log_path = os.path.join(RESULTS_DIR, f'rPCC_behavioral_accuracies_log_{n_sim}sim_noise{noise_levels}.txt')
        with open(behavior_log_path, 'w') as file_handle:
            file_handle.write(text_content)
        print(f'Saved {behavior_log_path}')
        # Save the figure
        fig.savefig(os.path.join(RESULTS_DIR, 'rPCC_pebd_Word_report_accuracy.pdf'), dpi=600)
        # Show the plot
        plt.show()



    """
    Figures 4.8, 4.16 (State Units r1, r2) & 4.10, 4.18 (Prediction Errors e1,e2,e1',e2')
    """

    sns.set_palette("colorblind")

    ### Correlations Between RDM's
    timeframe_list = [range(1,2), range(0,98)]


    if df_2 is not None:
        tstartg_gall = datetime.now()

        if use_cached and os.path.isfile(g_all_path):
            print(f"The file {g_all_path} exists. Loading it.")
            g_all = pd.read_pickle(g_all_path)
            g = df_2.set_index(["prime", "noise", "label", "timestep", "simulation"])
        else:
            print(f"The file {g_all_path} does not exist. Running calculations now.")
            g = df_2.set_index(["prime", "noise", "label", "timestep", "simulation"])
            g_all = pd.DataFrame()

            for timeframe in timeframe_list:
                df = g.query("timestep in {}".format(list(timeframe)))
                df = df.assign(timeframe=lambda x: "Time Step {}".format(timeframe[0]+1) if len(timeframe) <= 1 else "Time Steps {}-{}".format(timeframe[0]+1, timeframe[-1]+1))
                g_all = g_all.append(df)

            with open(g_all_path, "wb") as file_handle:
                pickle.dump(g_all, file_handle)
                print(f'Saved {g_all_path}')

            tendg_gall = datetime.now()
            print('tot time g_all prep', tendg_gall-tstartg_gall)


    """
    Bar plots r1,r2,e1,e2,e1',e2'
    """

    corr_method = "spearman"

    component_list = [
        "r1_hat",
        "r2_hat",
        "e10_abs",
        "e21_abs",
        "e11_abs",
        "e22_abs"
    ]

    component_list_rename = [
        "$\hat{r}_{1}(t)$",
        "$\hat{r}_{2}(t)$",
        "$e_{1}(t)$",
        "$e_{2}(t)$",
        "$e_{1}'(t)$",
        "$e_{2}'(t)$"
    ]

    component_rename_dict = dict(zip(component_list, component_list_rename))
    num_conditions = len(noise_levels) * len(prime_levels)
    noise_order = ['Low Clarity', 'High Clarity']

    '''
    check for pickled RSA lines (if not, run the RSA calculations)
    '''

    # Check if the file exists in submission
    if use_cached and os.path.isfile(rsa_lines_path):
        print(f"The file {rsa_lines_path} exists. Using it for RSA plots (lines and bars)")
        # load it with pickle
        with open(rsa_lines_path, "rb") as file_handle:
            all_rsa_allts_allsims = pickle.load(file_handle)
    else:
        print(f"The file {rsa_lines_path} does not exist, running RSA calculations now. 100 sims will take ~ 1.2 hours.")

        '''
        make lines
        '''

        # -------------------------------------------------------------
        # Make hRDM
        # -------------------------------------------------------------

        # Initialize matrix
        hrdm = np.ones((len(lexicon), len(lexicon)))

        for j, targ in enumerate(lexicon):
            for i, nontarg in enumerate(lexicon):
                # Check if the first two characters or the last two characters are the same
                if targ[:2] == nontarg[:2] or targ[-2:] == nontarg[-2:]:
                    # If they are, store a 0 in the matrix
                    hrdm[j, i] = 0

        # Masking, Compression for RSA
        # Create a mask for the upper triangle and main diagonal
        hmask = np.triu(np.ones_like(hrdm, dtype=bool))
        # Apply the mask to the array
        harray_masked = np.ma.masked_array(hrdm, mask=hmask)
        # Compress the masked array into a 1D array
        harray_compressed = harray_masked.compressed()

        # -------------------------------------------------------------
        # Calculation of RSA at each simulation (incls. each timestep for each component for each condition)
        # -------------------------------------------------------------

        num_conditions = len(noise_levels) * len(prime_levels)

        all_rsa_allts_allsims = {comp: [[] for _ in range(num_conditions)] for comp in component_list}

        for sim in range(n_sim):
            all_rsa_allts = {comp: [[] for _ in range(num_conditions)] for comp in component_list}

            for ts in range(98):
                all_rdms_tsi = {comp: [[] for _ in range(num_conditions)] for comp in component_list}

                for j, targ in enumerate(lexicon):
                    for i, nontarg in enumerate(lexicon):
                        for component in component_list:
                            condition_count = 0

                            for prime in prime_levels:
                                for noise in noise_levels:
                                    # Calculate dissimilarity
                                    dissim_targ_nontarg = 1 - pearsonr(g.loc[(prime, noise, targ, ts, sim), component], g.loc[(prime, noise, nontarg, ts, sim), component])[0]
                                    all_rdms_tsi[component][condition_count].append(dissim_targ_nontarg)

                                    condition_count += 1

                # This is RSA
                for component in component_list:
                    # At the component level, turn each of 4 condition RDMs into a numpy array
                    all_rdms_tsi[component] = [np.array(x).reshape(12,12) for x in all_rdms_tsi[component]]

                    # Calculate RSA for each condition
                    for condition in range(num_conditions):

                        # Create a mask for the upper triangle and main diagonal
                        mask = np.triu(np.ones_like(all_rdms_tsi[component][condition], dtype=bool))
                        # Apply the mask to the array
                        array_masked = np.ma.masked_array(all_rdms_tsi[component][condition], mask=mask)
                        # Compress the masked array into a 1D array
                        array_compressed = array_masked.compressed()

                        # Calculate RSA (Spearman) correlation (between our RDMs and the hypothesized RDM)
                        rsa_tsi = spearmanr(array_compressed, harray_compressed)[0]
                        # Store RSA values for each component
                        all_rsa_allts[component][condition].append(rsa_tsi)

            # add all_rsa_allts (per component, 4 lines (1 ea condition) of length 98) to 
            # all_rsa_allts_allsims (per component, 4 (x100) lines (100 ea condition) of length 98)
            for component in component_list:
                for condition in range(num_conditions):
                    all_rsa_allts_allsims[component][condition].append(all_rsa_allts[component][condition])        

        # Save the RSA lines
        with open(rsa_lines_path, "wb") as file_handle:
            pickle.dump(all_rsa_allts_allsims, file_handle)
            print(f'Saved {rsa_lines_path}')

    '''
    Univariate analysis: compute raw values (no RDM correlation) for e10_L1, e11_L1, r1_hat_L1, r1_unw_upd_L1
    '''

    univariate_component_list = ["e10_L1", "e11_L1", "r1_hat_L1", "r1_unw_upd_L1"]
    univariate_component_labels = {
        "e10_L1": "$e_{1}(t)$",
        "e11_L1": "$e'_{1}(t)$",
        "r1_hat_L1": "$\\hat{r}_{1}(t)$",
        "r1_unw_upd_L1": "$\\hat{r}_{1}(t)\ Upd.\ (unw.)$",
    }
    univariate_component_labels_ylabel = {
        "e10_L1": "$\\|e_{1}(t)\\|$",
        "e11_L1": "$\\|e'_{1}(t)\\|$",
        "r1_hat_L1": "$\\|\\hat{r}_{1}(t)\\|$",
        "r1_unw_upd_L1": "$\\|\\hat{r}_{1}(t)\\ \mathrm{Update\ term\ (unweighted)}\\|$",
    }

    # Create indexed dataframe for univariate calculations (needed whether cache is loaded or not)
    g = df_2.set_index(["prime", "noise", "label", "timestep", "simulation"])

    def compute_univariate_scalar(row: pd.Series, component: str) -> float:
        """Return the L1 magnitude of one univariate signal for a single trial row."""
        if component == "r1_hat_L1":
            return np.sum(np.abs(row["r1_hat"]))
        if component == "e10_L1":
            return np.sum(row["e10_abs"])
        if component == "e11_L1":
            return np.sum(row["e11_abs"])
        if component == "r1_unw_upd_L1":
            return np.sum(np.abs(row["r1_bu"] - row["r1_td"]))
        raise ValueError(f"Unknown univariate component: {component}")

    def compute_univariate_cache_components(
        component_subset: list[str],
        cache_dict: dict[str, list[list[list[float]]]],
    ) -> dict[str, list[list[list[float]]]]:
        """Fill `cache_dict` with per-sim, per-condition univariate time courses.

        For each component, stores `n_sim` lines of length 98 per condition,
        where each point is the lexicon-mean L1 magnitude at that timestep.
        """
        for comp in component_subset:
            cache_dict[comp] = [[] for _ in range(num_conditions)]

        for sim in range(n_sim):
            if sim % 10 == 0:
                print(f"Univariate line cache compute progress: sim {sim+1}/{n_sim}")
            all_univariate_allts = {comp: [[] for _ in range(num_conditions)] for comp in component_subset}

            for ts in range(98):
                for component in component_subset:
                    condition_count = 0
                    for prime in prime_levels:
                        for noise in noise_levels:
                            univ_values_across_words = []
                            for word in lexicon:
                                row = g.loc[(prime, noise, word, ts, sim)]
                                univ_value = compute_univariate_scalar(row, component)
                                univ_values_across_words.append(univ_value)
                            mean_univ_value = np.mean(univ_values_across_words)
                            all_univariate_allts[component][condition_count].append(mean_univ_value)
                            condition_count += 1

            for component in component_subset:
                for condition in range(num_conditions):
                    cache_dict[component][condition].append(all_univariate_allts[component][condition])

        return cache_dict

    # Check if the file exists in RESULTS_DIR
    if use_cached and os.path.isfile(univariate_lines_path):
        print(f"The file {univariate_lines_path} exists. Using it for Univariate plots")
        with open(univariate_lines_path, "rb") as file_handle:
            all_univariate_allts_allsims = pickle.load(file_handle)

        cache_valid = True
        for component in univariate_component_list:
            if component not in all_univariate_allts_allsims:
                cache_valid = False
                break

        if not cache_valid:
            print("Univariate cache schema is outdated (missing L1 components). Recomputing univariate lines.")
            all_univariate_allts_allsims = compute_univariate_cache_components(
                univariate_component_list,
                {}
            )
            with open(univariate_lines_path, "wb") as file_handle:
                pickle.dump(all_univariate_allts_allsims, file_handle)
                print(f'Saved {univariate_lines_path}')
    else:
        print(f"The file {univariate_lines_path} does not exist, running Univariate calculations now.")
        all_univariate_allts_allsims = compute_univariate_cache_components(
            univariate_component_list,
            {}
        )

        # Save the univariate lines
        with open(univariate_lines_path, "wb") as file_handle:
            pickle.dump(all_univariate_allts_allsims, file_handle)
            print(f'Saved {univariate_lines_path}')

    '''
    plot lines
    '''

    print('plotting lines starting at', datetime.now()-tstart)

    sns.set_palette("colorblind")
    colors = {'Neutral': sns.color_palette()[0], 'Match': sns.color_palette()[1]}
    noise_levels_as_words = ['Low Clarity', 'High Clarity']
    prime_levels_as_words = ['Neutral', 'Match']

    pretty_plot(labelsize=14, fontsize=14, lines=1.5, lfontsize=7, width=2)

    if show_lines:
        fig, axs = plt.subplots(6, 2, figsize=(8, 15.5))

        timesteps_to_mark = PLOT_TIMESTEP_ONSETS

        for i, component in enumerate(component_list):
            for j, noise in enumerate(noise_levels_as_words):
                ax = axs[i, j]
                for pi, prime in enumerate(prime_levels_as_words):
                    # Cache stored prime-outer/noise-inner: [LCN=0, HCN=1, LCM=2, HCM=3]
                    # Plot iterates noise-outer/prime-inner: LC→[0,2], HC→[1,3]
                    data_idx = [[0, 2], [1, 3]][j][pi]
                    # Convert the list of lists to a 2D numpy array
                    rsa_values = np.array(all_rsa_allts_allsims[component][data_idx])
                    # Create a DataFrame for the current condition
                    condition_data = pd.DataFrame(rsa_values.T, columns=['RSA'+str(k) for k in range(rsa_values.shape[0])])
                    condition_data['Time Step'] = range(1, rsa_values.shape[1] + 1)
                    condition_data = condition_data.melt(id_vars='Time Step', var_name='Simulation', value_name='RSA')
                    condition_data['Prime'] = prime
                    sns.lineplot(
                        x='Time Step',
                        y='RSA',
                        hue='Prime',
                        data=condition_data,
                        errorbar=errortype,
                        ax=ax,
                        palette=colors,
                    )


                # Phoneme onset markers (light grey, semi-transparent dashed)
                ax.set_xlim(*PLOT_TIMESTEP_XLIM)
                for t in timesteps_to_mark:
                    ax.axvline(x=t, color='0.75', alpha=0.75, linestyle='--', linewidth=1.75)

                ax.set_title(f'{component_list_rename[i]}  {noise}', fontsize=16, pad=9)

                # Position the legend in the upper right corner
                if i == 0 and j == 0:  # Only show legend for the first subplot
                    ax.legend(loc='upper left', title_fontsize='x-small', fontsize='x-small')
                else:
                    ax.legend().set_visible(False)

                # Only left column
                if j == 0:
                    ax.set_ylabel('RSA', fontsize=14)
                    if i == 2:
                        ax.yaxis.set_label_coords(-0.18, 0.5)
                    elif i == 4:
                        ax.yaxis.set_label_coords(-0.18, 0.5) 
                    elif i == 5:
                        ax.yaxis.set_label_coords(-0.18, 0.5)
                else:
                    ax.set_ylabel('')

                # Only bottom row
                if i == 5:
                    ax.set_xlabel('Time Step', fontsize=14)
                    ax.set_xticks(PLOT_TIMESTEP_TICKS)
                    ax.set_xticklabels(PLOT_TIMESTEP_TICKS)
                else:
                    ax.set_xlabel('')
                    ax.set_xticks(PLOT_TIMESTEP_TICKS)
                    ax.set_xticklabels([])

        plt.subplots_adjust(hspace=0.25)
        plt.subplots_adjust(wspace=0.5) 

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'rPCC_RSA_lines_{n_sim}sim_noise{noise_levels}.pdf'), dpi=600)
        plt.show()

    '''
    Univariate line plots (3 components × 2 noise levels)
    '''

    print('plotting univariate lines starting at', datetime.now()-tstart)

    sns.set_palette("colorblind")
    noise_levels_as_words = ['Low Clarity', 'High Clarity']
    prime_levels_as_words = ['Neutral', 'Match']
    # Use lighter versions of the colorblind palette (compatible with older seaborn)
    base_colorblind = sns.color_palette("colorblind", n_colors=2)
    light_mix = 0.45
    univariate_colors = {
        'Neutral': tuple(c + (1 - c) * light_mix for c in base_colorblind[0]),
        'Match': tuple(c + (1 - c) * light_mix for c in base_colorblind[1])
    }

    pretty_plot(labelsize=14, fontsize=14, lines=1.5, lfontsize=7, width=2)

    if show_lines:
        fig, axs = plt.subplots(4, 2, figsize=(8, 12.5))

        timesteps_to_mark = PLOT_TIMESTEP_ONSETS

        for i, component in enumerate(univariate_component_list):
            for j, noise in enumerate(noise_levels_as_words):
                ax = axs[i, j]
                for pi, prime in enumerate(prime_levels_as_words):
                    # Cache stored prime-outer/noise-inner: [LCN=0, HCN=1, LCM=2, HCM=3]
                    # Plot iterates noise-outer/prime-inner: LC→[0,2], HC→[1,3]
                    data_idx = [[0, 2], [1, 3]][j][pi]
                    # Convert the list of lists to a 2D numpy array
                    univ_values = np.array(all_univariate_allts_allsims[component][data_idx])
                    # Create a DataFrame for the current condition
                    condition_data = pd.DataFrame(univ_values.T, columns=['Univ'+str(k) for k in range(univ_values.shape[0])])
                    condition_data['Time Step'] = range(1, univ_values.shape[1] + 1)
                    condition_data = condition_data.melt(id_vars='Time Step', var_name='Simulation', value_name='Univariate')
                    condition_data['Prime'] = prime
                    sns.lineplot(
                        x='Time Step',
                        y='Univariate',
                        data=condition_data,
                        hue='Prime',
                        palette={prime: univariate_colors[prime]},
                        ax=ax,
                        alpha=0.3,
                        legend=(i == 0 and j == 0),
                    )

                # Phoneme onset markers (light grey, semi-transparent dashed)
                ax.set_xlim(*PLOT_TIMESTEP_XLIM)
                for t in timesteps_to_mark:
                    ax.axvline(x=t, color='0.75', alpha=0.75, linestyle='--', linewidth=1.75)

                ax.set_title(f'{univariate_component_labels[component]}  {noise}', fontsize=14, pad=9)

                # Position the legend in the upper right corner
                if i == 0 and j == 0:  # Only show legend for the first subplot
                    ax.legend(loc='upper left', title_fontsize='x-small', fontsize='x-small')
                else:
                    ax.legend().set_visible(False)

                # Only left column
                if j == 0:
                    ax.set_ylabel(univariate_component_labels_ylabel[component], fontsize=12)
                    if i == len(univariate_component_list) - 1:
                        ax.yaxis.set_label_coords(-0.18, 0.5)
                else:
                    ax.set_ylabel('')

                # Only bottom row
                if i == len(univariate_component_list) - 1:
                    ax.set_xlabel('Time Step', fontsize=14)
                    ax.set_xticks(PLOT_TIMESTEP_TICKS)
                    ax.set_xticklabels(PLOT_TIMESTEP_TICKS)
                else:
                    ax.set_xlabel('')
                    ax.set_xticks(PLOT_TIMESTEP_TICKS)
                    ax.set_xticklabels([])

        plt.subplots_adjust(hspace=0.25)
        plt.subplots_adjust(wspace=0.5) 

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'rPCC_Univariate_lines_{n_sim}sim_noise{noise_levels}.pdf'), dpi=600)
        plt.show()

    '''
    bars with cis
    '''

    dict_bars = {component:{'noise': ['Low Clarity', 'Low Clarity', 'High Clarity', 'High Clarity'],
        'prime': ['Neutral', 'Match', 'Neutral', 'Match'],
        'rsa': [43.45, 83.53, 79.17, 89.68],
        'ci': [(42.20, 44.70), (81.40, 85.66), (77.64, 80.70), (88.28, 90.08)]} for component in component_list}

    condition_counter_dict = {0: 'Low Clarity Neutral', 1: 'Low Clarity Match', 2: 'High Clarity Neutral', 3: 'High Clarity Match'}

    # Prepare data for seaborn, also print statements
    data = []
    bars_log_text = '\nRSA Bars (all Timesteps)\n\n'
    for i, component in enumerate(component_list):
        for condition_counter in range(4):
            # This is a 100x98 array (n = 100 sims, 98 timesteps each)
            # We want to average over all timesteps for each sim, creating 100 averages.
            # Then, we want to average over all sims, creating 1 average with 95% bootstrap CIs.
            # Cache stored prime-outer/noise-inner: [LCN=0, HCN=1, LCM=2, HCM=3]
            # dict_bars is noise-outer/prime-inner: [LCN=0, LCM=1, HCN=2, HCM=3]
            data_idx = [0, 2, 1, 3][condition_counter]
            rsa_values = np.array(all_rsa_allts_allsims[component][data_idx])
            # Calculate the mean of each sublist (average RSA over all 98 timesteps)
            rsa_means = np.nanmean(rsa_values, axis=1)

            cis_for_condition, mean_for_condition = compute_bootstrap_ci(rsa_means, stat='mean', nboot=10000, ci=0.95, return_stat=True)
            dict_bars[component]['rsa'][condition_counter] = mean_for_condition
            dict_bars[component]['ci'][condition_counter] = cis_for_condition
            comp_cond_text = f'component: {component}\ncondition: {condition_counter_dict[condition_counter]}\nmean_RSA_for_condition: {mean_for_condition}\ncis_for_condition: {cis_for_condition}'
            print(comp_cond_text)
            bars_log_text += comp_cond_text + '\n\n'

    with open(os.path.join(RESULTS_DIR, f'rPCC_RSA_bars_log_{n_sim}sim_noise{noise_levels}.txt'), 'w') as file_handle:
        file_handle.write(bars_log_text)

    plt.style.use('seaborn-paper')
    plt.rcParams['image.aspect'] = 'auto'
    plt.rcParams['figure.dpi'] = 600
    sns.set_palette("colorblind")

    # Create a figure with two subplots, side by side
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))

    # Assuming prime_levels is a list of prime levels for each bar in the left subplot
    color_dict = {'Neutral': sns.color_palette()[0], 'Match': sns.color_palette()[1]}
    prime_levels_as_words = ['Neutral', 'Match']
    colors_left = [color_dict[prime] for prime in prime_levels_as_words]

    axs_flat = axs.flatten()

    for c, component in enumerate(component_list):

        df_bars_component = pd.DataFrame(dict_bars[component])   

        # Plot the barplot
        barplot = sns.barplot(data=df_bars_component, x="noise", y="rsa", hue="prime", order=noise_order, errorbar=errortype, ax=axs_flat[c], palette=color_dict)

        x_values = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
        # RSA values
        y_values = [df_bars_component['rsa'][i] for i in range(4)]

        # Lower_errors and upper_errors are your arrays of errors
        all_condition_cis = [df_bars_component['ci'][i] for i in range(4)]

        all_condition_ci_err = []
        for i in range(4):
            errs = [abs(all_condition_cis[i][j]-y_values[i]) for j in range(2)]
            all_condition_ci_err.append(errs)

        lower_errors = [all_condition_ci_err[i][0] for i in range(4)]
        upper_errors = [all_condition_ci_err[i][1] for i in range(4)]

        # Careful to switch the middle two accuracies (LCM, HCN), 95% cis, around from their original order
        y_values = [y_values[0], y_values[2], y_values[1], y_values[3]]
        lower_errors = [lower_errors[0], lower_errors[2], lower_errors[1], lower_errors[3]]
        upper_errors = [upper_errors[0], upper_errors[2], upper_errors[1], upper_errors[3]]
        errors_95ci = [lower_errors, upper_errors]

        axs_flat[c].errorbar(x_values, y_values, yerr=errors_95ci, fmt='none', color='#444444', capsize=0, elinewidth=3.5, alpha=1)

        # Set the labels and legend for the subplot
        axs_flat[c].set_title(component_list_rename[c], fontsize=15, pad=9)
        if c % 2 == 0:
            if c == 4:
                axs_flat[c].set_ylabel('RSA', fontsize=14)
                axs_flat[c].yaxis.set_label_coords(-0.235, 0.5) 
            else:
                axs_flat[c].set_ylabel('RSA', fontsize=14)
        else:
            axs_flat[c].set_ylabel("")

        axs_flat[c].set_xticklabels(['Low Clarity', 'High Clarity'], fontsize=14)
        axs_flat[c].set_xlabel("")
        patches = [mpl.patches.Patch(color=sns.color_palette()[i], label=t) for i,t in enumerate(prime_levels_as_words)]
        axs_flat[c].legend(handles=patches, loc="upper left", fontsize="xx-small", title_fontsize="x-small")
        if c > 0:
            axs_flat[c].legend_.remove()

        # set y-axis limit
        all_condition_cis_arr = np.array(all_condition_cis)
        cis_min = all_condition_cis_arr.min()
        cis_max = all_condition_cis_arr.max()
        cis_ylims = np.array([cis_min, cis_max])

        y_values_arr = np.array(y_values)
        rsa_min = y_values_arr.min()
        rsa_max = y_values_arr.max()
        rsa_ylims = np.array([rsa_min, rsa_max])

        # Grab the proper ylim values, buffers dependent upon the sign
        # Check if any nans or infs in cis_ylims (first shot at ylims)
        if np.isnan(cis_ylims).any() or np.isinf(cis_ylims).any():
            # Grab the original RSA values
            if rsa_min < 0:
                low_bound = rsa_min * 1.1
            else:
                low_bound = rsa_min * 0.9

            if rsa_max < 0:
                high_bound = rsa_max * 0.9
            else:
                high_bound = rsa_max * 1.1

        else:
            if cis_min < 0:
                low_bound = cis_min * 1.1
            else:
                low_bound = cis_min * 0.9

            if cis_max < 0:
                high_bound = cis_max * 0.9
            else:
                high_bound = cis_max * 1.1

        if c > 0:
            ylims = [low_bound, high_bound]
            if c == 1:
                ylims = [low_bound, high_bound * 0.975]
            elif c == 3:
                # This is only in the final figures version- replace -0.002 with low_bound for universality
                ylims = [-0.002, high_bound]
        else:
            ylims = [low_bound, high_bound * 1.1]
        axs_flat[c].set_ylim(ylims)

        if c < 4:
            axs_flat[c].set_xticklabels([])

        axs_flat[c].tick_params(axis='y', labelsize=14)
        axs_flat[c].tick_params(axis='x', labelsize=14)
        axs_flat[c].set_title(component_list_rename[c], fontsize=17, pad=9)

    plt.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(wspace=0.5)   

    # Adjust the layout
    plt.tight_layout()
    # Save the figure
    fig.savefig(os.path.join(RESULTS_DIR, f'rPCC_RSA_bars_{n_sim}sim_noise{noise_levels}.pdf'), dpi=600)
    # Show the plot
    plt.show()

    '''
    Univariate bars with cis (all timesteps)
    '''

    dict_univariate_bars = {component:{'noise': ['Low Clarity', 'Low Clarity', 'High Clarity', 'High Clarity'],
        'prime': ['Neutral', 'Match', 'Neutral', 'Match'],
        'mean': [0.0, 0.0, 0.0, 0.0],
        'ci': [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]} for component in univariate_component_list}

    condition_counter_dict = {0: 'Low Clarity Neutral', 1: 'Low Clarity Match', 2: 'High Clarity Neutral', 3: 'High Clarity Match'}

    # Prepare data for seaborn, also print statements
    data = []
    univariate_bars_log_text = '\nUnivariate Bars (all Timesteps)\n\n'
    for i, component in enumerate(univariate_component_list):
        for condition_counter in range(4):
            # This is a 100x98 array (n = 100 sims, 98 timesteps each)
            # We want to average over all timesteps for each sim, creating 100 averages.
            # Then, we want to average over all sims, creating 1 average with 95% bootstrap CIs.
            # Cache stored prime-outer/noise-inner: [LCN=0, HCN=1, LCM=2, HCM=3]
            # dict_univariate_bars is noise-outer/prime-inner: [LCN=0, LCM=1, HCN=2, HCM=3]
            data_idx = [0, 2, 1, 3][condition_counter]
            univ_values = np.array(all_univariate_allts_allsims[component][data_idx])
            # Calculate the mean of each sublist (average univariate value over all 98 timesteps)
            univ_means = np.nanmean(univ_values, axis=1)

            cis_for_condition, mean_for_condition = compute_bootstrap_ci(univ_means, stat='mean', nboot=10000, ci=0.95, return_stat=True)
            dict_univariate_bars[component]['mean'][condition_counter] = mean_for_condition
            dict_univariate_bars[component]['ci'][condition_counter] = cis_for_condition
            comp_cond_text = f'component: {component}\ncondition: {condition_counter_dict[condition_counter]}\nmean_for_condition: {mean_for_condition}\ncis_for_condition: {cis_for_condition}'
            print(comp_cond_text)
            univariate_bars_log_text += comp_cond_text + '\n\n'

    with open(os.path.join(RESULTS_DIR, f'rPCC_Univariate_bars_log_{n_sim}sim_noise{noise_levels}.txt'), 'w') as file_handle:
        file_handle.write(univariate_bars_log_text)

    plt.style.use('seaborn-paper')
    plt.rcParams['image.aspect'] = 'auto'
    plt.rcParams['figure.dpi'] = 600
    sns.set_palette("colorblind")

    # Create a figure with one column (4 rows for 4 components)
    fig, axs = plt.subplots(4, 1, figsize=(4, 10))

    # Reuse the univariate light colors from the line plot section
    univariate_color_dict = univariate_colors.copy()
    prime_levels_as_words = ['Neutral', 'Match']
    colors_left = [univariate_color_dict[prime] for prime in prime_levels_as_words]

    axs_flat = axs.flatten()

    for c, component in enumerate(univariate_component_list):

        df_bars_component = pd.DataFrame(dict_univariate_bars[component])   

        # Plot the barplot
        barplot = sns.barplot(data=df_bars_component, x="noise", y="mean", hue="prime", order=noise_order, errorbar=errortype, ax=axs_flat[c], palette=univariate_color_dict)

        x_values = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
        # Univariate values
        y_values = [df_bars_component['mean'][i] for i in range(4)]

        # Lower_errors and upper_errors are your arrays of errors
        all_condition_cis = [df_bars_component['ci'][i] for i in range(4)]

        all_condition_ci_err = []
        for i in range(4):
            errs = [abs(all_condition_cis[i][j]-y_values[i]) for j in range(2)]
            all_condition_ci_err.append(errs)

        lower_errors = [all_condition_ci_err[i][0] for i in range(4)]
        upper_errors = [all_condition_ci_err[i][1] for i in range(4)]

        # Careful to switch the middle two accuracies (LCM, HCN), 95% cis, around from their original order
        y_values = [y_values[0], y_values[2], y_values[1], y_values[3]]
        lower_errors = [lower_errors[0], lower_errors[2], lower_errors[1], lower_errors[3]]
        upper_errors = [upper_errors[0], upper_errors[2], upper_errors[1], upper_errors[3]]
        errors_95ci = [lower_errors, upper_errors]

        axs_flat[c].errorbar(x_values, y_values, yerr=errors_95ci, fmt='none', color='#444444', capsize=0, elinewidth=3.5, alpha=1)

        axs_flat[c].set_ylabel(univariate_component_labels_ylabel[component], fontsize=12)
        axs_flat[c].set_xlabel('')
        if c == len(univariate_component_list) - 1:
            axs_flat[c].set_xticklabels(['Low Clarity', 'High Clarity'], fontsize=14)
        else:
            axs_flat[c].set_xticklabels([])
        if c == 0:
            axs_flat[c].legend(loc='upper right', fontsize='small')
        else:
            axs_flat[c].legend().set_visible(False)

    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.5)   

    # Adjust the layout
    plt.tight_layout()
    # Save the figure
    fig.savefig(os.path.join(RESULTS_DIR, f'rPCC_Univariate_bars_{n_sim}sim_noise{noise_levels}.pdf'), dpi=600)
    # Show the plot
    plt.show()

    '''
    ts2 bars with cis
    '''

    dict_bars_ts2 = {component:{'noise': ['Low Clarity', 'Low Clarity', 'High Clarity', 'High Clarity'],
        'prime': ['Neutral', 'Match', 'Neutral', 'Match'],
        'rsa': [43.45, 83.53, 79.17, 89.68],
        'ci': [(42.20, 44.70), (81.40, 85.66), (77.64, 80.70), (88.28, 90.08)]} for component in component_list}

    condition_counter_dict = {0: 'Low Clarity Neutral', 1: 'Low Clarity Match', 2: 'High Clarity Neutral', 3: 'High Clarity Match'}

    # Prepare data for seaborn, also print statements
    data = []
    bars_ts2_log_text = '\nRSA Bars (Timestep 2 only)\n\n'
    for i, component in enumerate(component_list):
        for condition_counter in range(4):
            # This is a 100x98 array (n = 100 sims, 98 timesteps each)
            # We want to average over all timesteps for each sim, creating 100 averages.
            # Then, we want to average over all sims, creating 1 average with 95% bootstrap CIs.
            # Cache stored prime-outer/noise-inner: [LCN=0, HCN=1, LCM=2, HCM=3]
            # dict_bars_ts2 is noise-outer/prime-inner: [LCN=0, LCM=1, HCN=2, HCM=3]
            data_idx = [0, 2, 1, 3][condition_counter]
            rsa_values = np.array(all_rsa_allts_allsims[component][data_idx])
            # Calculate the avg RSA at TS2 (index 1)
            rsa_ts2s = rsa_values[:,1]

            cis_for_condition, ts2_for_condition = compute_bootstrap_ci(rsa_ts2s, stat='mean', nboot=10000, ci=0.95, return_stat=True)
            dict_bars_ts2[component]['rsa'][condition_counter] = ts2_for_condition
            dict_bars_ts2[component]['ci'][condition_counter] = cis_for_condition
            comp_cond_text = f'component: {component}\ncondition: {condition_counter_dict[condition_counter]}\navg_ts2_RSA_for_condition: {ts2_for_condition}\ncis_for_condition: {cis_for_condition}'
            print(comp_cond_text)
            bars_ts2_log_text += comp_cond_text + '\n\n'

    with open(os.path.join(RESULTS_DIR, f'rPCC_RSA_bars_ts2_log_{n_sim}sim_noise{noise_levels}.txt'), 'w') as file_handle:
        file_handle.write(bars_ts2_log_text)

    plt.style.use('seaborn-paper')
    plt.rcParams['image.aspect'] = 'auto'
    plt.rcParams['figure.dpi'] = 600
    sns.set_palette("colorblind")

    # Create a figure with two subplots, side by side
    fig, axs = plt.subplots(3, 2, figsize=(8, 8))

    # Assuming prime_levels is a list of prime levels for each bar in the left subplot
    color_dict = {'Neutral': sns.color_palette()[0], 'Match': sns.color_palette()[1]}
    prime_levels_as_words = ['Neutral', 'Match']
    colors_left = [color_dict[prime] for prime in prime_levels_as_words]

    axs_flat = axs.flatten()

    for c, component in enumerate(component_list):

        df_bars_component = pd.DataFrame(dict_bars_ts2[component])   

        # Plot the barplot
        barplot = sns.barplot(data=df_bars_component, x="noise", y="rsa", hue="prime", order=noise_order, errorbar=errortype, ax=axs_flat[c], palette=color_dict)

        x_values = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
        # RSA values
        y_values = [df_bars_component['rsa'][i] for i in range(4)]

        # Lower_errors and upper_errors are your arrays of errors
        all_condition_cis = [df_bars_component['ci'][i] for i in range(4)]

        all_condition_ci_err = []
        for i in range(4):
            errs = [abs(all_condition_cis[i][j]-y_values[i]) for j in range(2)]
            all_condition_ci_err.append(errs)

        lower_errors = [all_condition_ci_err[i][0] for i in range(4)]
        upper_errors = [all_condition_ci_err[i][1] for i in range(4)]

        # Careful to switch the middle two accuracies (LCM, HCN), 95% cis, around from their original order
        y_values = [y_values[0], y_values[2], y_values[1], y_values[3]]
        lower_errors = [lower_errors[0], lower_errors[2], lower_errors[1], lower_errors[3]]
        upper_errors = [upper_errors[0], upper_errors[2], upper_errors[1], upper_errors[3]]
        errors_95ci = [lower_errors, upper_errors]

        axs_flat[c].errorbar(x_values, y_values, yerr=errors_95ci, fmt='none', color='#444444', capsize=0, elinewidth=3.5, alpha=1)

        # Set the labels and legend for the subplot
        axs_flat[c].set_title(component_list_rename[c], fontsize=15, pad=9)
        if c % 2 == 0:
            axs_flat[c].set_ylabel('RSA', fontsize=14)
            if c == 0:
                axs_flat[c].yaxis.set_label_coords(-0.235, 0.5)
            elif c == 4:
                axs_flat[c].yaxis.set_label_coords(-0.215, 0.5)
        else:
            axs_flat[c].set_ylabel("")

        axs_flat[c].set_xticklabels(['Low Clarity', 'High Clarity'], fontsize=14)
        axs_flat[c].set_xlabel("")
        patches = [mpl.patches.Patch(color=sns.color_palette()[i], label=t) for i,t in enumerate(prime_levels_as_words)]
        axs_flat[c].legend(handles=patches, loc="upper right", fontsize="xx-small", title_fontsize="x-small")
        if c > 0:
            axs_flat[c].legend_.remove()

        # set y-axis limit
        all_condition_cis_arr = np.array(all_condition_cis)
        cis_min = all_condition_cis_arr.min()
        cis_max = all_condition_cis_arr.max()
        cis_ylims = np.array([cis_min, cis_max])

        y_values_arr = np.array(y_values)
        rsa_min = y_values_arr.min()
        rsa_max = y_values_arr.max()
        rsa_ylims = np.array([rsa_min, rsa_max])

        # Grab the proper ylim values, buffers dependent upon the sign
        # Check if any nans or infs in cis_ylims (first shot at ylims)
        if np.isnan(cis_ylims).any() or np.isinf(cis_ylims).any():
            # Grab the original RSA values
            if rsa_min < 0:
                low_bound = rsa_min * 1.1
            else:
                low_bound = rsa_min * 0.9

            if rsa_max < 0:
                high_bound = rsa_max * 0.9
            else:
                high_bound = rsa_max * 1.1

        else:
            if cis_min < 0:
                low_bound = cis_min * 1.1
            else:
                low_bound = cis_min * 0.9

            if cis_max < 0:
                high_bound = cis_max * 0.9
            else:
                high_bound = cis_max * 1.1

        if c > 0:
            ylims = [low_bound, high_bound]
            if c == 1:
                ylims = [low_bound, high_bound * 0.975]
        else:
            ylims = [low_bound, high_bound * 1.1]

        axs_flat[c].set_ylim(ylims)

        if c < 4:
            axs_flat[c].set_xticklabels([])

        axs_flat[c].tick_params(axis='y', labelsize=14)
        axs_flat[c].tick_params(axis='x', labelsize=14)
        axs_flat[c].set_title(component_list_rename[c], fontsize=17, pad=9)

    plt.subplots_adjust(hspace=0.25)
    plt.subplots_adjust(wspace=0.5)   

    # Adjust the layout
    plt.tight_layout()
    # Save the figure
    fig.savefig(os.path.join(RESULTS_DIR, f'rPCC_RSA_bars_ts2_{n_sim}sim_noise{noise_levels}.pdf'), dpi=600)
    # Show the plot
    plt.show()

    '''
    Univariate ts2 bars with cis (timestep 2 only)
    '''

    dict_univariate_bars_ts2 = {component:{'noise': ['Low Clarity', 'Low Clarity', 'High Clarity', 'High Clarity'],
        'prime': ['Neutral', 'Match', 'Neutral', 'Match'],
        'mean': [0.0, 0.0, 0.0, 0.0],
        'ci': [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]} for component in univariate_component_list}

    condition_counter_dict = {0: 'Low Clarity Neutral', 1: 'Low Clarity Match', 2: 'High Clarity Neutral', 3: 'High Clarity Match'}

    # Prepare data for seaborn, also print statements
    data = []
    univariate_bars_ts2_log_text = '\nUnivariate Bars (Timestep 2 only)\n\n'
    for i, component in enumerate(univariate_component_list):
        for condition_counter in range(4):
            # This is a 100x98 array (n = 100 sims, 98 timesteps each)
            # Filter to timestep 2 (index 1), creating 100x1
            # Average across the single timestep to get 1 value per sim (which is just the ts2 value)
            # Cache stored prime-outer/noise-inner: [LCN=0, HCN=1, LCM=2, HCM=3]
            # dict_univariate_bars_ts2 is noise-outer/prime-inner: [LCN=0, LCM=1, HCN=2, HCM=3]
            data_idx = [0, 2, 1, 3][condition_counter]
            univ_values = np.array(all_univariate_allts_allsims[component][data_idx])
            # Extract only timestep 2 (index 1) for each sim
            univ_ts2_values = univ_values[:, 1]

            cis_for_condition, mean_for_condition = compute_bootstrap_ci(univ_ts2_values, stat='mean', nboot=10000, ci=0.95, return_stat=True)
            dict_univariate_bars_ts2[component]['mean'][condition_counter] = mean_for_condition
            dict_univariate_bars_ts2[component]['ci'][condition_counter] = cis_for_condition
            comp_cond_text = f'component: {component}\ncondition: {condition_counter_dict[condition_counter]}\nmean_for_condition: {mean_for_condition}\ncis_for_condition: {cis_for_condition}'
            print(comp_cond_text)
            univariate_bars_ts2_log_text += comp_cond_text + '\n\n'

    with open(os.path.join(RESULTS_DIR, f'rPCC_Univariate_bars_ts2_log_{n_sim}sim_noise{noise_levels}.txt'), 'w') as file_handle:
        file_handle.write(univariate_bars_ts2_log_text)

    plt.style.use('seaborn-paper')
    plt.rcParams['image.aspect'] = 'auto'
    plt.rcParams['figure.dpi'] = 600
    sns.set_palette("colorblind")

    # Create a figure with one column (4 rows for 4 components)
    fig, axs = plt.subplots(4, 1, figsize=(4, 10))

    # Reuse the univariate light colors from the line plot section
    univariate_color_dict = univariate_colors.copy()
    prime_levels_as_words = ['Neutral', 'Match']
    colors_left = [univariate_color_dict[prime] for prime in prime_levels_as_words]

    axs_flat = axs.flatten()

    for c, component in enumerate(univariate_component_list):

        df_bars_component = pd.DataFrame(dict_univariate_bars_ts2[component])   

        # Plot the barplot
        barplot = sns.barplot(data=df_bars_component, x="noise", y="mean", hue="prime", order=noise_order, errorbar=errortype, ax=axs_flat[c], palette=univariate_color_dict)

        x_values = [patch.get_x() + patch.get_width() / 2 for patch in barplot.patches]
        # Univariate values
        y_values = [df_bars_component['mean'][i] for i in range(4)]

        # Lower_errors and upper_errors are your arrays of errors
        all_condition_cis = [df_bars_component['ci'][i] for i in range(4)]

        all_condition_ci_err = []
        for i in range(4):
            errs = [abs(all_condition_cis[i][j]-y_values[i]) for j in range(2)]
            all_condition_ci_err.append(errs)

        lower_errors = [all_condition_ci_err[i][0] for i in range(4)]
        upper_errors = [all_condition_ci_err[i][1] for i in range(4)]

        # Careful to switch the middle two accuracies (LCM, HCN), 95% cis, around from their original order
        y_values = [y_values[0], y_values[2], y_values[1], y_values[3]]
        lower_errors = [lower_errors[0], lower_errors[2], lower_errors[1], lower_errors[3]]
        upper_errors = [upper_errors[0], upper_errors[2], upper_errors[1], upper_errors[3]]
        errors_95ci = [lower_errors, upper_errors]

        axs_flat[c].errorbar(x_values, y_values, yerr=errors_95ci, fmt='none', color='#444444', capsize=0, elinewidth=3.5, alpha=1)

        axs_flat[c].set_ylabel(univariate_component_labels_ylabel[component], fontsize=12)
        axs_flat[c].set_xlabel('')
        if c == len(univariate_component_list) - 1:
            axs_flat[c].set_xticklabels(['Low Clarity', 'High Clarity'], fontsize=14)
        else:
            axs_flat[c].set_xticklabels([])
        if c == 0:
            axs_flat[c].legend(loc='upper right', fontsize='small')
        else:
            axs_flat[c].legend().set_visible(False)

    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0.5)   

    # Adjust the layout
    plt.tight_layout()
    # Save the figure
    fig.savefig(os.path.join(RESULTS_DIR, f'rPCC_Univariate_bars_ts2_{n_sim}sim_noise{noise_levels}.pdf'), dpi=600)
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
