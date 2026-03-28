
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# ── Zerve Design System ──────────────────────────────────────────────────────
_BG        = "#1D1D20"
_TEXT_PRI  = "#fbfbff"
_TEXT_SEC  = "#909094"
_BLUE      = "#A1C9F4"    # non-success
_ORANGE    = "#FFB482"    # success
_GREEN     = "#8DE5A1"
_YELLOW    = "#ffd400"
_CORAL     = "#FF9F9B"
_GRID_COL  = "#2e2e33"

_LABEL_MAP = {
    "total_events":       "Total Events",
    "active_days":        "Active Days",
    "total_sessions":     "Total Sessions",
    "events_per_day":     "Events / Day",
    "run_block_count":    "Run Block Count",
    "agent_usage_count":  "Agent Usage Count",
    "blocks_created":     "Blocks Created",
    "credits_used_total": "Credits Used",
    "unique_tools_used":  "Unique Tools Used",
    "events_per_session": "Events / Session",
}

plt.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor":   _BG,
    "axes.edgecolor":   _TEXT_SEC,
    "axes.labelcolor":  _TEXT_PRI,
    "text.color":       _TEXT_PRI,
    "xtick.color":      _TEXT_SEC,
    "ytick.color":      _TEXT_SEC,
    "grid.color":       _GRID_COL,
    "font.family":      "sans-serif",
    "font.size":        10,
})

# ── Build cohort dataframe ───────────────────────────────────────────────────
_cohort_df = X.copy()
_cohort_df["success"] = y.values

_success_df = _cohort_df[_cohort_df["success"] == 1].drop(columns="success")
_fail_df    = _cohort_df[_cohort_df["success"] == 0].drop(columns="success")
_feat_list  = list(X.columns)

_n_success  = int(y.sum())
_n_total    = len(y)
_success_rate = _n_success / _n_total

print(f"Cohort: {_n_total:,} users  |  Success: {_n_success} ({_success_rate:.2%})  |  Non-Success: {_n_total - _n_success:,}")
print(f"Note: Extreme class imbalance. Activation thresholds use relative probability tiers.")
print()

# ── Discriminability: Cohen's d (glass delta variant) ───────────────────────
_discriminability = {}
for _feat in _feat_list:
    _mu0 = _fail_df[_feat].mean()
    _mu1 = _success_df[_feat].mean()
    _sd0 = _fail_df[_feat].std() + 1e-9
    _discriminability[_feat] = abs(_mu1 - _mu0) / _sd0   # Glass's Δ

_top6 = sorted(_discriminability, key=_discriminability.get, reverse=True)[:6]
print("Top 6 most discriminative features (Glass's Δ — effect size vs non-success SD):")
for _f in _top6:
    print(f"  {_LABEL_MAP.get(_f, _f):<28}  Δ = {_discriminability[_f]:.3f}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Violin (non-success) + jittered scatter/box (success) for all features
# ─────────────────────────────────────────────────────────────────────────────
_ncols = 2
_nrows = (len(_feat_list) + 1) // _ncols

violin_distribution_chart = plt.figure(figsize=(16, _nrows * 3.5))
violin_distribution_chart.patch.set_facecolor(_BG)
violin_distribution_chart.suptitle(
    "Feature Distribution: Successful vs Non-Successful Users",
    fontsize=16, fontweight="bold", color=_TEXT_PRI, y=1.01
)

_rng = np.random.default_rng(42)

for _i, _feat in enumerate(_feat_list):
    _ax = violin_distribution_chart.add_subplot(_nrows, _ncols, _i + 1)
    _ax.set_facecolor(_BG)

    _p99 = float(np.percentile(_cohort_df[_feat], 99))
    _p1  = float(np.percentile(_cohort_df[_feat], 1))
    _data_fail    = _fail_df[_feat].clip(_p1, _p99).values
    _data_success = _success_df[_feat].clip(_p1, _p99).values

    # Violin for non-success (large group)
    _vp = _ax.violinplot([_data_fail], positions=[0], showmedians=True,
                          showextrema=False, widths=0.65)
    for _pc in _vp["bodies"]:
        _pc.set_facecolor(_BLUE); _pc.set_alpha(0.60); _pc.set_edgecolor(_BLUE)
    _vp["cmedians"].set_color(_BLUE); _vp["cmedians"].set_linewidth(2.5)
    _ax.scatter([0], [_data_fail.mean()], color=_BLUE, s=55, zorder=5,
                marker="D", edgecolors="white", linewidths=0.8)

    # Success cohort: box + individual jittered points (more honest for n=8)
    if len(_data_success) >= 2:
        _q25, _q50, _q75 = np.percentile(_data_success, [25, 50, 75])
        _iqr_s = _q75 - _q25
        _w = 0.14
        _ax.add_patch(plt.Rectangle((1 - _w, _q25), 2 * _w, max(_iqr_s, 0.01),
                                     facecolor=_ORANGE, alpha=0.40,
                                     edgecolor=_ORANGE, lw=1.5))
        _wlo = max(_data_success.min(), _q25 - 1.5 * _iqr_s)
        _whi = min(_data_success.max(), _q75 + 1.5 * _iqr_s)
        _ax.plot([1, 1], [_wlo, _q25], color=_ORANGE, lw=1.5, alpha=0.8)
        _ax.plot([1, 1], [_q75, _whi], color=_ORANGE, lw=1.5, alpha=0.8)
        _ax.plot([1 - _w, 1 + _w], [_q50, _q50], color=_ORANGE, lw=2.5, zorder=6)
        _jitter = _rng.uniform(-0.07, 0.07, len(_data_success))
        _ax.scatter(1 + _jitter, _data_success, color=_ORANGE, s=45,
                    alpha=0.80, zorder=7, edgecolors="none")
    elif len(_data_success) == 1:
        _ax.scatter([1], _data_success, color=_ORANGE, s=100, zorder=7, marker="*")

    _label = _LABEL_MAP.get(_feat, _feat)
    _ax.set_title(_label, fontsize=11, fontweight="bold", color=_TEXT_PRI, pad=6)
    _ax.set_xticks([0, 1])
    _ax.set_xticklabels(
        [f"Non-Success\n(n={len(_data_fail):,})", f"Success\n(n={len(_data_success)})"],
        fontsize=8.5, color=_TEXT_PRI
    )
    _ax.tick_params(axis="y", labelcolor=_TEXT_SEC, labelsize=8)
    _ax.grid(axis="y", alpha=0.25)
    _ax.spines[["top", "right"]].set_visible(False)

    _med_fail = np.median(_data_fail)
    _med_succ = np.median(_data_success)
    _pct_diff = ((_med_succ - _med_fail) / (_med_fail + 1e-9)) * 100
    _sign = "+" if _pct_diff >= 0 else ""
    _ax.text(0.5, 0.96, f"Δ median: {_sign}{_pct_diff:.0f}%",
             ha="center", va="top", transform=_ax.transAxes,
             fontsize=8.5, color=_YELLOW, fontweight="bold")

_legend_elems = [
    mpatches.Patch(facecolor=_BLUE, alpha=0.70, label="Non-Success (violin + mean ◆)"),
    mpatches.Patch(facecolor=_ORANGE, alpha=0.55, label="Success (box + raw points)"),
]
violin_distribution_chart.legend(
    handles=_legend_elems, loc="upper right", ncol=2,
    facecolor=_BG, edgecolor=_TEXT_SEC, labelcolor=_TEXT_PRI,
    fontsize=10, bbox_to_anchor=(1.0, 1.04)
)
plt.tight_layout(rect=[0, 0, 1, 1])
print("✅ Violin distribution chart created")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Overlapping histograms — top 6 features (no KDE to avoid singularity)
# ─────────────────────────────────────────────────────────────────────────────
histogram_comparison_chart = plt.figure(figsize=(16, 10))
histogram_comparison_chart.patch.set_facecolor(_BG)
histogram_comparison_chart.suptitle(
    "Top 6 Discriminative Features — Density-Normalised Distribution Comparison",
    fontsize=14, fontweight="bold", color=_TEXT_PRI, y=1.01
)

for _i, _feat in enumerate(_top6):
    _ax = histogram_comparison_chart.add_subplot(2, 3, _i + 1)
    _ax.set_facecolor(_BG)

    _p99 = float(np.percentile(_cohort_df[_feat], 99))
    _p1  = float(np.percentile(_cohort_df[_feat], 1))
    _bins = np.linspace(_p1, _p99, 40)

    _ax.hist(_fail_df[_feat].clip(_p1, _p99), bins=_bins,
             color=_BLUE, alpha=0.50, density=True, label="Non-Success")

    # Success histogram — only if enough distinct values; else rug plot
    _succ_vals = _success_df[_feat].clip(_p1, _p99).values
    _n_unique = len(np.unique(_succ_vals))
    if _n_unique >= 2:
        _ax.hist(_succ_vals, bins=min(10, _n_unique), range=(_p1, _p99),
                 color=_ORANGE, alpha=0.65, density=True, label="Success")
    else:
        # Rug plot for degenerate case
        for _sv in _succ_vals:
            _ax.axvline(_sv, color=_ORANGE, alpha=0.5, lw=1.5)
        _ax.plot([], [], color=_ORANGE, lw=1.5, label="Success (rug)")

    _m0 = np.median(_fail_df[_feat])
    _m1 = np.median(_success_df[_feat])
    _ax.axvline(_m0, color=_BLUE, lw=2, linestyle="--", alpha=0.9,
                label=f"Non-Success  med={_m0:.1f}")
    _ax.axvline(_m1, color=_ORANGE, lw=2, linestyle="--", alpha=0.9,
                label=f"Success  med={_m1:.1f}")

    _label = _LABEL_MAP.get(_feat, _feat)
    _ax.set_title(_label, fontsize=11, fontweight="bold", color=_TEXT_PRI, pad=6)
    _ax.set_xlabel("Value (clipped p1–p99)", fontsize=9, color=_TEXT_SEC)
    _ax.set_ylabel("Density", fontsize=9, color=_TEXT_SEC)
    _ax.tick_params(labelsize=8, colors=_TEXT_SEC)
    _ax.grid(axis="y", alpha=0.2)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(fontsize=7.5, facecolor=_BG, edgecolor=_TEXT_SEC,
               labelcolor=_TEXT_PRI, loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 1])
print("✅ Overlapping histogram chart created")


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION THRESHOLD ANALYSIS
# Due to extreme class imbalance (0.15%), model probabilities are tiny.
# Thresholds = percentiles of model output on the full held-out test set:
#   P50 = median risk tier, P75 = high-risk tier, P90 = elite zone
# ─────────────────────────────────────────────────────────────────────────────
_quantile_grid = np.linspace(0.01, 0.99, 300)
_medians       = X.median()

_all_probs_test = best_model.predict_proba(X_test)[:, 1]
_prob_p50 = float(np.percentile(_all_probs_test, 50))
_prob_p75 = float(np.percentile(_all_probs_test, 75))
_prob_p90 = float(np.percentile(_all_probs_test, 90))

print(f"Model P(success) range on test set:")
print(f"  Min={_all_probs_test.min():.4f}  Max={_all_probs_test.max():.4f}")
print(f"  P50={_prob_p50:.4f}  P75={_prob_p75:.4f}  P90={_prob_p90:.4f}")
print()

_rel_thresholds = {
    "P50 (median)": _prob_p50,
    "P75 (high)":   _prob_p75,
    "P90 (elite)":  _prob_p90,
}
_thr_colors = {
    "P50 (median)": _GREEN,
    "P75 (high)":   _YELLOW,
    "P90 (elite)":  _CORAL,
}

activation_thresholds = {}
_summary_rows = []

for _feat in _feat_list:
    _sweep_vals = np.quantile(X[_feat], _quantile_grid)
    _base_row   = _medians.copy()
    _rows = []
    for _sv in _sweep_vals:
        _base_row[_feat] = _sv
        _rows.append(_base_row.values.copy())
    _sweep_df = pd.DataFrame(_rows, columns=_feat_list)
    _probs = best_model.predict_proba(_sweep_df)[:, 1]

    _crossings = {}
    for _thr_name, _thr_val in _rel_thresholds.items():
        _idx = np.where(_probs >= _thr_val)[0]
        if len(_idx) > 0:
            _crossings[_thr_name] = {
                "quantile": _quantile_grid[_idx[0]],
                "value":    _sweep_vals[_idx[0]],
                "prob":     _thr_val,
            }
        else:
            _crossings[_thr_name] = None

    activation_thresholds[_feat] = {
        "crossings":  _crossings,
        "probs":      _probs,
        "sweep_vals": _sweep_vals,
    }

    _label = _LABEL_MAP.get(_feat, _feat)
    _row   = {"feature": _feat, "feature_label": _label}
    for _tname, _c in _crossings.items():
        _short = _tname.split()[0]
        _row[f"value_{_short}"]    = round(_c["value"], 3) if _c else None
        _row[f"quantile_{_short}"] = round(_c["quantile"] * 100, 1) if _c else None
    _summary_rows.append(_row)

activation_summary_df = pd.DataFrame(_summary_rows)


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Probability curves with crossing annotations (top 6 features)
# ─────────────────────────────────────────────────────────────────────────────
probability_curves_chart = plt.figure(figsize=(16, 10))
probability_curves_chart.patch.set_facecolor(_BG)
probability_curves_chart.suptitle(
    "Activation Threshold Analysis — P(Success) vs Feature Quantile\n"
    "(Other features at median. Thresholds = P50/P75/P90 of model output distribution.)",
    fontsize=13, fontweight="bold", color=_TEXT_PRI, y=1.03
)

for _i, _feat in enumerate(_top6):
    _ax = probability_curves_chart.add_subplot(2, 3, _i + 1)
    _ax.set_facecolor(_BG)

    _probs_sweep = activation_thresholds[_feat]["probs"]

    _ax.plot(_quantile_grid * 100, _probs_sweep * 100,
             color=_BLUE, lw=2.5, label="P(Success)")
    _ax.fill_between(_quantile_grid * 100, _probs_sweep * 100,
                     alpha=0.12, color=_BLUE)

    for _thr_name, _thr_val in _rel_thresholds.items():
        _col = _thr_colors[_thr_name]
        _ax.axhline(_thr_val * 100, color=_col, lw=1.3,
                    linestyle="--", alpha=0.85,
                    label=f"{_thr_name}: {_thr_val*100:.2f}%")
        _cross = activation_thresholds[_feat]["crossings"].get(_thr_name)
        if _cross:
            _ax.scatter([_cross["quantile"] * 100], [_thr_val * 100],
                        color=_col, s=90, zorder=6, edgecolors="white", lw=0.8)
            _ax.annotate(
                f" {_cross['value']:.1f}",
                xy=(_cross["quantile"] * 100, _thr_val * 100),
                fontsize=7, color=_col, va="bottom",
            )

    _label = _LABEL_MAP.get(_feat, _feat)
    _ax.set_title(_label, fontsize=11, fontweight="bold", color=_TEXT_PRI, pad=6)
    _ax.set_xlabel("Feature Quantile (%)", fontsize=9, color=_TEXT_SEC)
    _ax.set_ylabel("P(Success) %", fontsize=9, color=_TEXT_SEC)
    _ax.set_xlim(0, 100)
    _ax.tick_params(labelsize=8, colors=_TEXT_SEC)
    _ax.grid(axis="both", alpha=0.2)
    _ax.spines[["top", "right"]].set_visible(False)
    _ax.legend(fontsize=7, facecolor=_BG, edgecolor=_TEXT_SEC,
               labelcolor=_TEXT_PRI, loc="upper left", handlelength=1.2)

plt.tight_layout(rect=[0, 0, 1, 1])
print("✅ Probability curves chart created")


# ─────────────────────────────────────────────────────────────────────────────
# Print Activation Threshold Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 82)
print("  ACTIVATION THRESHOLD SUMMARY")
print(f"  P50={_prob_p50*100:.3f}%  |  P75={_prob_p75*100:.3f}%  |  P90={_prob_p90*100:.3f}%  (absolute P(success))")
print("=" * 82)
print(f"  {'Feature':<24}  {'Value @ P50':>14} {'Pctile':>7}  {'Value @ P75':>14} {'Pctile':>7}  {'Value @ P90':>14} {'Pctile':>7}")
print("  " + "-" * 80)

for _row in _summary_rows:
    _parts = []
    for _short in ["P50", "P75", "P90"]:
        _v = _row.get(f"value_{_short}")
        _q = _row.get(f"quantile_{_short}")
        if _v is not None:
            _parts.append(f"{_v:>14.2f} {_q:>6.1f}%")
        else:
            _parts.append(f"{'N/A':>14} {'N/A':>7}")
    print(f"  {_row['feature_label']:<24}  " + "  ".join(_parts))

print()
print("=" * 82)
print(f"\n✅ Exported: activation_summary_df ({len(activation_summary_df)} features × {len(activation_summary_df.columns)} cols)")
