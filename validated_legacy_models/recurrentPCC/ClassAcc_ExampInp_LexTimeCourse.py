"""Classification-accuracy, example-input, and lexical time-course plots for the rPCC.

Load trained recurrent predictive-coding classifier (rPCC) runs and produce
training- and testing-protocol classification-accuracy curves with bootstrap
CIs, an example phonological-feature input image, and lexical-activation
time-course plots (after and during training). Validated legacy reference code.
"""
from __future__ import annotations

from typing import Set
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import string

script_dir = Path(__file__).resolve().parent
repo_root = Path(__file__).resolve().parents[2]
for path in (script_dir, repo_root):
	if str(path) not in sys.path:
		sys.path.insert(0, str(path))

from validated_legacy_models.recurrentPCC.rPCC_Model import Model, softmax, category_given_target
from Utils import add_vertical_timestep_lines, compute_bootstrap_ci, load_trained_namespace

from kbutil.plotting import pylab_pretty_plot as pretty_plot

PLOT_TIMESTEP_TICKS = [1, 20, 40, 60, 80, 100]
PLOT_TIMESTEP_ONSETS = [2, 21, 40, 59]
PLOT_TIMESTEP_XLIM = (-1, 100)
RESULTS_DIR = script_dir / "models_plus_results" / "rPCC_cvcv12_train"
RAW_CVCV12_DIR = script_dir / "data" / "raw" / "CVCV_12"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def timestep_series_for_plot(series: pd.Series) -> pd.Series:
	"""Return a 1-indexed timestep series for plotting.

	If the incoming timesteps are 0-indexed (min==0), shift by +1.
	"""
	try:
		min_val = int(np.nanmin(series.to_numpy()))
	except Exception:
		return series
	return series + 1 if min_val == 0 else series


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
		# Use a literal baseline caret (not an accent). Do this last so we don't
		# escape the braces we introduce here.
		.replace('^', r'{\char94}')
	)


def _build_unicode_italic_map() -> dict[str, str]:
	"""Precompute unicode math italic glyphs for ASCII letters."""
	italic_map = {}
	lower_start = 0x1D44E
	upper_start = 0x1D434
	for idx, ch in enumerate(string.ascii_lowercase):
		italic_map[ch] = chr(lower_start + idx)
	for idx, ch in enumerate(string.ascii_uppercase):
		italic_map[ch] = chr(upper_start + idx)
	return italic_map


_UNICODE_ITALIC_MAP = _build_unicode_italic_map()



def _italicize_indices(text: str, indices: Set[int]) -> str:
	"""Return `text` with character positions in `indices` italicized via unicode."""
	if not indices:
		return text
	chars = list(text)
	for idx in indices:
		if 0 <= idx < len(chars):
			replacement = _UNICODE_ITALIC_MAP.get(chars[idx])
			if replacement:
				chars[idx] = replacement
	return ''.join(chars)


def _segment_text_by_style(text: str, indices: Set[int]) -> list[tuple[str, bool]]:
	"""Yield (substring, italic_flag) tuples for contiguous runs."""
	if not text:
		return []
	segments = []
	current_chars = []
	current_flag = None
	for idx, ch in enumerate(text):
		flag = idx in indices
		if current_flag is None:
			current_flag = flag
			current_chars.append(ch)
		elif flag == current_flag:
			current_chars.append(ch)
		else:
			segments.append((''.join(current_chars), current_flag))
			current_chars = [ch]
			current_flag = flag
	if current_chars:
		segments.append((''.join(current_chars), current_flag))
	return segments


def format_r2_tick_label(label: str, target_word: str, *, usetex: bool) -> str:
	"""Apply prefix/suffix italicization rules for lexical activation plots."""
	if not target_word:
		return latex_safe_text(label) if usetex else label
	italic_indices: Set[int] = set()
	if label == target_word:
		italic_indices = set(range(len(label)))
	else:
		if len(label) >= 2 and len(target_word) >= 2:
			if label[:2] == target_word[:2]:
				italic_indices.update({0, 1})
			if label[-2:] == target_word[-2:]:
				italic_indices.update({len(label) - 2, len(label) - 1})
	if not usetex:
		return _italicize_indices(label, italic_indices)
	segments = _segment_text_by_style(label, italic_indices)
	parts = []
	for segment_text, italic_flag in segments:
		escaped_segment = latex_safe_text(segment_text)
		if italic_flag:
			parts.append(r"\textbf{" + escaped_segment + "}")
		else:
			parts.append(escaped_segment)
	return ''.join(parts)


# -------------------------------------------------------------
# Matplotlib settings
# -------------------------------------------------------------


def _apply_mpl_defaults() -> None:
	"""Reset Matplotlib to the shared defaults used by every plot here."""
	mpl.rcdefaults()
	plt.rcParams['image.aspect'] = 'auto'
	plt.rcParams['figure.dpi'] = 600


_PROTOCOL_SPECS = {
	"training": {
		"df_key": "output_train_df",
		"title": "Training",
		"note": (
			"Training protocol: accuracy is scored on r2_hat states recorded during the training pass itself.\n"
			"The correct one-hot label is clamped on the top layer while the network infers each word, and\n"
			"weights update word-by-word within the epoch, so each scored state reflects both label guidance\n"
			"and partial mid-epoch learning.\n"
		),
	},
	"testing": {
		"df_key": "output_test_df",
		"title": "Testing",
		"note": (
			"Testing protocol: accuracy is scored on a separate evaluation pass at each checkpoint, run before\n"
			"that epoch's training. Weights are frozen and the label input is zeroed, so classification reflects\n"
			"only what the learned weights extract from the input alone.\n"
		),
	},
}


def plot_classification_accuracy(num_training_runs: int, *, protocol: str = "training", nboot: int = 10000) -> None:
	"""Plot mean classification accuracy over training runs for a given protocol.

	Args:
		num_training_runs: Number of trained_rpcc_X.pkl runs to load.
		protocol: 'training' (mid-epoch, label-clamped states) or 'testing'
			(frozen-weight, zero-label evaluation pass at each checkpoint).
		nboot: Bootstrap resamples per checkpoint for the 95% CIs.
	"""
	spec = _PROTOCOL_SPECS[protocol]
	_apply_mpl_defaults()
	pretty_plot(width=2, lines=1.5)

	all_run_dfs = []
	epoch_max_val = None
	save_interval_val = None

	for run in range(num_training_runs):
		pkl_path = str(RESULTS_DIR / f"trained_rpcc_{run+1}.pkl")
		ns = load_trained_namespace(pkl_path)

		epoch_max_val = max(range(ns["epoch_n"]))
		save_interval_val = ns["save_interval"]

		run_df = ns[spec["df_key"]].copy()
		run_df[f"epoch_adj_{run+1}"] = run_df.epoch + 1
		run_df[f"accuracy_percent_{run+1}"] = run_df.accuracy * 100
		all_run_dfs.append(run_df)

	# Concatenate all run dataframes into one, col-wise
	all_run_dfs = pd.concat(all_run_dfs, axis=1)
	# remove all columns which don't contain "epoch_adj" or "accuracy_percent"
	all_run_dfs = all_run_dfs.loc[:, all_run_dfs.columns.str.contains("epoch_adj|accuracy_percent")]
	# Grab all unique epochs where checkpoints were saved
	checkpoint_nums = all_run_dfs.epoch_adj_1.unique()

	# loop through checkpoints for mean accuracy measurements, 95% CIs
	accs = []
	low_bounds = []
	high_bounds = []

	text_content = f'Mean {protocol} accuracy over {num_training_runs} training runs, by checkpoint\n'
	text_content += spec["note"]
	text_content += f'95% CIs calculated using {nboot} bootstraps each checkpoint\n'
	text_content += f'{len(checkpoint_nums)} checkpoints: {len(checkpoint_nums)-1} over {epoch_max_val+1} epochs, saved every {save_interval_val} eps, + 1 (epoch 1)\n\n'
	print(text_content)

	for cpt, checkpoint in enumerate(checkpoint_nums):
		word_accs_checkpt = all_run_dfs.query("epoch_adj_1 == @checkpoint").filter(regex="accuracy_percent")
		mean_accs_checkpt = word_accs_checkpt.mean(axis=0)
		mean_acc_checkpt = mean_accs_checkpt.mean(axis=0)
		mean_accs_checkpt_arr = mean_accs_checkpt.to_numpy()
		checkpt_cis = compute_bootstrap_ci(mean_accs_checkpt_arr, stat='mean', nboot=nboot, ci=0.95, return_stat=False)
		checkpt_errs = [abs(checkpt_cis[bound] - mean_acc_checkpt) for bound in [0, 1]]
		results = (
			f"Checkpoint {cpt} Epoch {checkpoint}: {mean_acc_checkpt:.2f} "
			f"lowCI: {checkpt_cis[0]:.2f} highCI: {checkpt_cis[1]:.2f} "
			f"lowErr: {checkpt_errs[0]:.2f} highErr: {checkpt_errs[1]:.2f}"
		)
		text_content += results + '\n'
		print(results)
		accs.append(mean_acc_checkpt)
		low_bounds.append(checkpt_cis[0])
		high_bounds.append(checkpt_cis[1])

	accs_df = pd.DataFrame({"checkpoint": checkpoint_nums, "acc": accs, "low_bound": low_bounds, "high_bound": high_bounds})
	artifact_stem = f'rPCC_class_acc_{protocol}'
	with open(RESULTS_DIR / f'{artifact_stem}.txt', 'w') as f:
		f.write(text_content)

	fig = plt.figure(figsize=(4, 3))
	sns.lineplot(data=accs_df, x="checkpoint", y="acc", err_style=None)
	plt.fill_between(accs_df["checkpoint"], accs_df["low_bound"], accs_df["high_bound"], color='b', alpha=.3)
	plt.xlabel("Epoch", fontsize=11)
	plt.ylabel(r"Mean accuracy (\%)", fontsize=11)
	plt.title(spec["title"], fontsize=11)
	plt.xticks([0, 2500, 5000], [1, 2500, 5000], fontsize=11)
	plt.yticks(fontsize=11)
	plt.ylim(0, 109)
	plt.tight_layout()
	fig.savefig(RESULTS_DIR / f'{artifact_stem}.pdf', dpi=600)
	plt.show()


def plot_example_input(*, example_path: str = str(RAW_CVCV12_DIR / "003_k^ra.png"), timesteps_to_mark: list[int] | None = None) -> None:
	"""Plot the example phonological feature input image."""
	_apply_mpl_defaults()
	pretty_plot(width=2)

	if timesteps_to_mark is None:
		timesteps_to_mark = PLOT_TIMESTEP_ONSETS

	example_path = Path(example_path)
	if example_path.suffix.lower() == ".pdf":
		png_fallback = example_path.with_suffix(".png")
		if png_fallback.exists():
			example_path = png_fallback
	if not example_path.exists():
		raise FileNotFoundError(f"Example input image not found at {example_path}")
	I = cv2.imread(str(example_path))
	if I is None:
		raise ValueError(f"OpenCV could not read example input image at {example_path}")
	I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
	I = I / 255.0
	I = I.astype(np.float32)

	ncols = 1
	nrows = 1
	subplot_x, subplot_y = (5, 3)

	feature_list = [
		'alveolar', 'anterior', 'back', 'central', 'consonantal',
		'continuant', 'coronal', 'high', 'labial', 'low',
		'mid', 'nasal', 'palatal', 'reduced', 'round',
		'sibilant', 'sonorant', 'syllabic', 'tense', 'velar', 'voiced'
	]

	fig, ax = plt.subplots(
		nrows, ncols,
		figsize=(subplot_x * ncols, subplot_y * nrows),
		sharex="all", sharey="all",
		constrained_layout=True,
	)

	# Plot against 1-indexed timesteps to match interpretation of onsets
	n_timesteps = I.shape[1]
	im = ax.imshow(
		I,
		cmap="bone_r",
		vmin=0,
		vmax=1,
		extent=(0.5, n_timesteps + 0.5, I.shape[0] - 0.5, -0.5),
	)
	ax.set_title(r'/k $\hat{  }$ ra/', fontsize=14, pad=9)
	ax.tick_params(axis='y', pad=0.1, left=False)
	ax.set_yticks(range(len(feature_list)))
	ax.set_yticklabels(feature_list, fontsize=9.5)
	plt.xticks(fontsize=11)
	ax.set_xlabel("Time Step", fontsize=11)

	# Set the x-ticks/labels based on the actual number of timesteps
	desired_tick_labels = [1, 20, 40, 60, 80, 100]
	tick_labels = [t for t in desired_tick_labels if t <= n_timesteps]
	if not tick_labels or tick_labels[-1] != n_timesteps:
		tick_labels.append(n_timesteps)
	ax.set_xticks(tick_labels)
	ax.set_xticklabels(tick_labels)

	add_vertical_timestep_lines(ax, timesteps_to_mark)

	cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
	cbar.ax.tick_params(labelsize=10)
	cbar.outline.set_linewidth(2)
	fig.savefig(RESULTS_DIR / 'Example_input_k^ra.pdf', dpi=600)
	plt.show()


def plot_lexical_activation_plots(*, nboot: int = 10000, timesteps_to_mark: list[int] | None = None) -> None:
	"""Plot lexical activation timecourses after and during training."""
	_apply_mpl_defaults()
	pretty_plot(width=2, lines=1.5)

	if timesteps_to_mark is None:
		timesteps_to_mark = [2, 21, 40, 59]

	# Load Trained rPCC and Parameters/Hyperparameters (single run)
	pkl_path = str(RESULTS_DIR / "trained_rpcc_1.pkl")
	ns = load_trained_namespace(pkl_path)

	output_test_df = ns["output_test_df"]

	# Restrict lexical activation averaging to correct test trials only.
	# Note: For the original 12-item test set this is usually all rows, but we
	# guard for datasets where some trials are incorrect.
	output_test_df = output_test_df.query("accuracy == 1").reset_index(drop=True)

	output_train_df = ns["output_train_df"]
	I_dict = ns["I_dict"]
	softmax_c = ns["softmax_c"]
	cohort_len = ns["cohort_len"]
	rhyme_len = ns["rhyme_len"]

	epoch_max = max(range(ns["epoch_n"]))
	I_size = output_test_df.I.iloc[0].size
	L_size = len(I_dict)
	r2_size = L_size

	# Construct model for completeness (not strictly required for plotting)
	model = Model(I_size, ns["r1_size"], r2_size, L_size, seed=ns["weight_init_seed"])
	model.s10 = ns["s10"]
	model.s11 = ns["s11"]
	model.s21 = ns["s21"]
	model.s22 = ns["s22"]
	model.s32 = ns["s32"]
	model.alpha_1 = ns["alpha_1"]
	model.alpha_2 = ns["alpha_2"]
	model.beta_1 = ns["beta_1"]
	model.beta_2 = ns["beta_2"]
	model.gamma_1 = ns["gamma_1"]
	model.gamma_2 = ns["gamma_2"]
	model.U1 = ns["all_weights_df"].U1.iloc[-1]
	model.U2 = ns["all_weights_df"].U2.iloc[-1]
	model.V1 = ns["all_weights_df"].V1.iloc[-1]
	model.V2 = ns["all_weights_df"].V2.iloc[-1]

	output_test_df["epoch_adj"] = output_test_df.epoch + 1
	output_train_df["epoch_adj"] = output_train_df.epoch + 1
	output_train_df["accuracy_percent"] = output_train_df.accuracy * 100

	# Color palette
	palette = sns.color_palette('colorblind', n_colors=4)
	np.random.seed(316)
	np.random.shuffle(palette)
	# Reset to bootstrap seed so seaborn bands match classification CI seed
	np.random.seed(123)
	colors = [tuple(c) for c in palette]
	blue = colors[0]
	orange = colors[1]
	yellow = colors[3]
	green = colors[2]
	colors = [blue, orange, yellow, green]

	vline_xs = list(timesteps_to_mark)
	ylabel_lat = r"Lexical activation"

	# After training
	r2_ss = output_test_df.query("epoch == @epoch_max").set_index(["label", "timestep"]).r2_hat
	r2_df = r2_ss.apply(lambda x: pd.Series(softmax(x, c=softmax_c)).rename(lambda x: list(I_dict.keys())[x])).rename_axis("node", axis=1)
	r2_df_long = r2_df.melt(ignore_index=False).reset_index(level=["label", "timestep"])
	r2_df_long["category"] = r2_df_long.apply(lambda x: category_given_target(x.label, x.node, cohort_len=cohort_len, rhyme_len=rhyme_len), axis=1)
	r2_df_long["timestep_plot"] = timestep_series_for_plot(r2_df_long["timestep"])

	fig = plt.figure(figsize=(4, 3))
	g = sns.lineplot(
		x="timestep_plot",
		y="value",
		hue="category",
		hue_order=["target", "cohort", "rhyme", "other"],
		style="category",
		style_order=["target", "cohort", "other", "rhyme"],
		err_style="band",
		n_boot=nboot,
		data=r2_df_long,
		palette=colors,
	)

	add_vertical_timestep_lines(g, vline_xs)
	g.set_xlim(*PLOT_TIMESTEP_XLIM)
	g.set_xticks(PLOT_TIMESTEP_TICKS)
	g.set_xticklabels(PLOT_TIMESTEP_TICKS)

	leg = g.legend(labels=["Target", "Cohort", "Rhyme", "Unrelated"], loc="upper left", fontsize=8)
	plt.xticks(fontsize=11)
	plt.yticks(fontsize=11)
	g.set_xlabel("Time Step", fontsize=11)
	g.set_ylabel(ylabel_lat, fontsize=11)

	# Add word labels aligned with legend rows, positioned at timestep ~41
	fig.canvas.draw()
	renderer = fig.canvas.get_renderer()
	usetex = mpl.rcParams.get('text.usetex', False)
	if usetex:
		kibu_label = r"kibu \textit{(e.g.)}"
	else:
		kibu_label = r"kibu $\it{(e.g.)}$"
	legend_words = [
		latex_safe_text("k^ra"),
		latex_safe_text("k^si"),
		latex_safe_text("b^ra"),
		kibu_label,
	]
	legend_texts = leg.get_texts()[:len(legend_words)]
	blend = mpl.transforms.blended_transform_factory(g.transData, g.transAxes)
	y_positions = []
	for text_artist in legend_texts:
		bbox = text_artist.get_window_extent(renderer=renderer)
		y_center_display = 0.5 * (bbox.y0 + bbox.y1)
		y_axes = g.transAxes.inverted().transform((0, y_center_display))[1]
		y_positions.append(y_axes)

	# Make the stack a bit looser + slightly lower
	y_shift = -0.03
	spacing_scale = 1.25
	center_y = float(np.mean(y_positions))
	y_positions = [center_y + (y - center_y) * spacing_scale + y_shift for y in y_positions]

	# Keep the first three labels centered at x=41, but left-align the final
	# "e.g. kibu" so it is flush with the left edge of the other example words.
	reference_text = g.text(41, y_positions[0], legend_words[0], transform=blend, ha='center', va='center', fontsize=8, color='black', alpha=0)
	reference_bbox = reference_text.get_window_extent(renderer=renderer)
	reference_text.remove()
	left_edge_data = g.transData.inverted().transform((reference_bbox.x0, 0))[0]

	for idx, (word, y_axes) in enumerate(zip(legend_words, y_positions)):
		if idx == len(legend_words) - 1:
			g.text(left_edge_data, y_axes, word, transform=blend, ha='left', va='center', fontsize=8, color='black')
		else:
			g.text(41, y_axes, word, transform=blend, ha='center', va='center', fontsize=8, color='black')

	plt.tight_layout()
	fig.savefig(RESULTS_DIR / 'rPCC_lex_act_after_train.pdf', dpi=600)
	plt.show()

	# During training: Epoch 200-1000
	plot_interval = 200
	plot_number = 5
	ncols = 5
	nrows = plot_number // ncols + int(bool(plot_number % ncols))
	subplot_x, subplot_y = (2.5, 2.5)
	fig, axes = plt.subplots(
		nrows,
		ncols,
		figsize=(subplot_x * ncols, subplot_y * nrows),
		sharex=True,
		sharey=True,
		constrained_layout=True,
	)
	for i, (e, ax) in enumerate(zip(range(plot_interval, plot_interval * plot_number + 1, plot_interval), axes)):
		r2_ss = output_test_df.query("epoch_adj == @e").set_index(["label", "timestep"]).r2_hat
		r2_df = r2_ss.apply(lambda x: pd.Series(softmax(x, c=softmax_c)).rename(lambda x: list(I_dict.keys())[x])).rename_axis("node", axis=1)
		r2_df_long = r2_df.melt(ignore_index=False).reset_index(level=["label", "timestep"])
		r2_df_long["category"] = r2_df_long.apply(lambda x: category_given_target(x.label, x.node, cohort_len=cohort_len, rhyme_len=rhyme_len), axis=1)
		r2_df_long["timestep_plot"] = timestep_series_for_plot(r2_df_long["timestep"])
		g = sns.lineplot(
			x="timestep_plot",
			y="value",
			hue="category",
			hue_order=["target", "cohort", "rhyme", "other"],
			style="category",
			style_order=["target", "cohort", "other", "rhyme"],
			err_style="band",
			data=r2_df_long,
			ax=ax,
			palette=colors,
		)
		add_vertical_timestep_lines(ax, vline_xs)
		ax.set_xlim(*PLOT_TIMESTEP_XLIM)
		ax.set_xticks(PLOT_TIMESTEP_TICKS)
		ax.set_xticklabels(PLOT_TIMESTEP_TICKS)
		if i == 0:
			g.legend(labels=["Target", "Cohort", "Rhyme", "Other"], loc="upper left", fontsize='xx-small')
		else:
			ax.get_legend().remove()
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		g.set_xlabel("Time Step", fontsize=12)
		g.set_ylabel(ylabel_lat, fontsize=12)
		ax.set_title("Epoch {}".format(e))
	fig.savefig(RESULTS_DIR / 'rPCC_lex_act_during_trainEp_200-1000.pdf', dpi=600)
	plt.show()

	# During training: Epoch 1000-5000
	plot_interval = 1000
	plot_number = 5
	ncols = 5
	nrows = plot_number // ncols + int(bool(plot_number % ncols))
	subplot_x, subplot_y = (2.5, 2.5)
	fig, axes = plt.subplots(
		nrows,
		ncols,
		figsize=(subplot_x * ncols, subplot_y * nrows),
		sharex=True,
		sharey=True,
		constrained_layout=True,
	)
	for i, (e, ax) in enumerate(zip(range(plot_interval, plot_interval * plot_number + 1, plot_interval), axes)):
		r2_ss = output_test_df.query("epoch_adj == @e").set_index(["label", "timestep"]).r2_hat
		r2_df = r2_ss.apply(lambda x: pd.Series(softmax(x, c=softmax_c)).rename(lambda x: list(I_dict.keys())[x])).rename_axis("node", axis=1)
		r2_df_long = r2_df.melt(ignore_index=False).reset_index(level=["label", "timestep"])
		r2_df_long["category"] = r2_df_long.apply(lambda x: category_given_target(x.label, x.node, cohort_len=cohort_len, rhyme_len=rhyme_len), axis=1)
		r2_df_long["timestep_plot"] = timestep_series_for_plot(r2_df_long["timestep"])
		g = sns.lineplot(
			x="timestep_plot",
			y="value",
			hue="category",
			hue_order=["target", "cohort", "rhyme", "other"],
			style="category",
			style_order=["target", "cohort", "other", "rhyme"],
			err_style="band",
			data=r2_df_long,
			ax=ax,
			palette=colors,
		)
		add_vertical_timestep_lines(ax, vline_xs)
		ax.set_xlim(*PLOT_TIMESTEP_XLIM)
		ax.set_xticks(PLOT_TIMESTEP_TICKS)
		ax.set_xticklabels(PLOT_TIMESTEP_TICKS)
		if i == 0:
			g.legend(labels=["Target", "Cohort", "Rhyme", "Other"], loc="upper left", fontsize='xx-small')
		else:
			ax.get_legend().remove()
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		g.set_xlabel("Time Step", fontsize=12)
		g.set_ylabel(ylabel_lat, fontsize=12)
		ax.set_title("Epoch {}".format(e))
	fig.savefig(RESULTS_DIR / 'rPCC_lex_act_during_trainEp_1000-5000.pdf', dpi=600)
	plt.show()


def main(
	num_training_runs: int = 1,
	*,
	print_train_classification_accuracy: bool = True,
	print_test_classification_accuracy: bool = True,
	print_examp_input: bool = True,
	print_lexical_activation_plots: bool = True,
) -> None:
	"""Run plots with optional toggles."""
	if print_train_classification_accuracy:
		plot_classification_accuracy(num_training_runs, protocol="training")
	if print_test_classification_accuracy:
		plot_classification_accuracy(num_training_runs, protocol="testing")
	if print_examp_input:
		plot_example_input()
	if print_lexical_activation_plots:
		plot_lexical_activation_plots()


if __name__ == "__main__":
	main(
		num_training_runs=1, # This will have to be at 100 for the full classification accuracy plot (requires 100 separate trained_rpcc_X.pkl files)
		print_train_classification_accuracy=True,
		print_test_classification_accuracy=True,
		print_examp_input=True,
		print_lexical_activation_plots=True,
	)


