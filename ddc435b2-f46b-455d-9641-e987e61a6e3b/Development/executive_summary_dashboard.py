import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import warnings
warnings.filterwarnings("ignore")

# ── Zerve Design System ─────────────────────────────────────────────────────
_BG        = "#1D1D20"
_TEXT_PRI  = "#fbfbff"
_TEXT_SEC  = "#909094"
_BLUE      = "#A1C9F4"
_ORANGE    = "#FFB482"
_GREEN     = "#8DE5A1"
_YELLOW    = "#ffd400"
_LAVENDER  = "#D0BBFF"
_CORAL     = "#FF9F9B"
_GRID      = "#2e2e33"
_PANEL_BG  = "#25252a"

plt.rcParams.update({
    "figure.facecolor":  _BG,
    "axes.facecolor":    _BG,
    "axes.edgecolor":    _GRID,
    "axes.labelcolor":   _TEXT_PRI,
    "text.color":        _TEXT_PRI,
    "xtick.color":       _TEXT_SEC,
    "ytick.color":       _TEXT_SEC,
    "grid.color":        _GRID,
    "font.family":       "sans-serif",
    "font.size":         10,
})

# ──────────────────────────────────────────────────────────────────────────────
# PANEL A — MODEL PERFORMANCE: ROC Curve + 4 key metrics
# ──────────────────────────────────────────────────────────────────────────────
_y_prob_exec = best_model.predict_proba(X_test)[:, 1]
_fpr_exec, _tpr_exec, _ = roc_curve(y_test, _y_prob_exec)
_auc_exec = auc(_fpr_exec, _tpr_exec)
_auc_pr_exec = average_precision_score(y_test, _y_prob_exec)
_prec_exec, _rec_exec, _ = precision_recall_curve(y_test, _y_prob_exec)

# ──────────────────────────────────────────────────────────────────────────────
# PANEL B — FEATURE IMPORTANCE (top 7 features by MDI)
# ──────────────────────────────────────────────────────────────────────────────
_feat_names = list(X_test.columns)
_importances = best_model.feature_importances_
_sorted_fi = np.argsort(_importances)  # ascending
_label_map_exec = {
    "total_events":       "Total Events",
    "active_days":        "Active Days",
    "total_sessions":     "Total Sessions",
    "events_per_day":     "Events / Day",
    "run_block_count":    "Run Block Count",
    "agent_usage_count":  "Agent Usage",
    "blocks_created":     "Blocks Created",
    "credits_used_total": "Credits Used",
    "unique_tools_used":  "Unique Tools",
    "events_per_session": "Events / Session",
}
_sorted_names_fi = [_label_map_exec.get(_feat_names[i], _feat_names[i]) for i in _sorted_fi]
_sorted_vals_fi  = _importances[_sorted_fi]
_fi_colors = [_YELLOW if i >= len(_sorted_fi) - 3 else _BLUE for i in range(len(_sorted_fi))]

# ──────────────────────────────────────────────────────────────────────────────
# PANEL C — SEGMENT SUCCESS RATES
# ──────────────────────────────────────────────────────────────────────────────
_seg_names_exec  = ["🚀 Power Users", "🔍 Casual Explorers", "⚠️ At-Risk Users"]
_seg_success_exec = [11.6, 0.0, 0.0]   # % values from kmeans output
_seg_n_exec       = [69, 618, 4723]
_seg_colors_exec  = [_GREEN, _BLUE, _ORANGE]

# ──────────────────────────────────────────────────────────────────────────────
# PANEL D — ACTIVATION THRESHOLDS  
# Activation thresholds from business_driver_analysis (actual computed values)
# ──────────────────────────────────────────────────────────────────────────────
_thr_features_exec = ["Credits Used\n(≥ 50)", "Run Blocks\n(≥ 20)", "Unique Tools\n(≥ 7)", "Agent Usage\n(≥ 1)"]
_thr_sr_exec       = [100.0, 8.1, 12.5, 1.1]   # Success rates at each threshold (from block output)
_thr_lift_exec     = [676, 55, 85, 7]            # Lift values  
_thr_colors_exec   = [_YELLOW, _GREEN, _LAVENDER, _ORANGE]
_baseline_exec     = 0.148   # 0.148%

# ──────────────────────────────────────────────────────────────────────────────
# BUILD THE 2×2 FIGURE
# ──────────────────────────────────────────────────────────────────────────────
summary_dashboard = plt.figure(figsize=(18, 14))
summary_dashboard.patch.set_facecolor(_BG)

# Super title
summary_dashboard.text(
    0.5, 0.975,
    "Zerve User Success Intelligence — Executive Summary Dashboard",
    ha="center", va="top", fontsize=18, fontweight="bold",
    color=_TEXT_PRI, transform=summary_dashboard.transFigure,
)
summary_dashboard.text(
    0.5, 0.955,
    "Hackathon Analysis  ·  5,410 Users  ·  RandomForest  ·  AUC-ROC = 1.0000  ·  Decision Threshold = 0.116",
    ha="center", va="top", fontsize=11, color=_TEXT_SEC,
    transform=summary_dashboard.transFigure,
)

gs = gridspec.GridSpec(2, 2, figure=summary_dashboard,
                       left=0.07, right=0.97, top=0.94, bottom=0.06,
                       hspace=0.38, wspace=0.30)

# ── Panel A: ROC Curve ───────────────────────────────────────────────────────
ax_a = summary_dashboard.add_subplot(gs[0, 0])
ax_a.plot(_fpr_exec, _tpr_exec, color=_BLUE, lw=2.8, label=f"ROC (AUC = {_auc_exec:.4f})")
ax_a.plot([0, 1], [0, 1], color=_GRID, lw=1.5, linestyle="--", label="Random Baseline")
ax_a.fill_between(_fpr_exec, _tpr_exec, alpha=0.10, color=_BLUE)
# PR curve as second line on same axes
_ax_a2 = ax_a.twinx()
_ax_a2.set_facecolor(_BG)
_ax_a2.plot(_rec_exec, _prec_exec, color=_ORANGE, lw=2.0, linestyle="-",
            alpha=0.75, label=f"PR  (AP = {_auc_pr_exec:.4f})")
_ax_a2.set_ylabel("Precision", color=_ORANGE, fontsize=9)
_ax_a2.tick_params(axis="y", colors=_ORANGE, labelsize=8)
_ax_a2.set_ylim(-0.01, 1.05)
_ax_a2.spines["right"].set_color(_ORANGE)
_ax_a2.spines["top"].set_visible(False)

ax_a.set_xlabel("False Positive Rate  |  Recall →", fontsize=10)
ax_a.set_ylabel("True Positive Rate", fontsize=10)
ax_a.set_title("A  ·  Model Performance", fontsize=13, fontweight="bold",
               color=_TEXT_PRI, loc="left", pad=10)
ax_a.set_xlim([-0.01, 1.01])
ax_a.set_ylim([-0.01, 1.05])
ax_a.grid(True, alpha=0.25)
ax_a.spines["top"].set_visible(False)

# Key metrics as text boxes in corner
_metrics_text = (
    f"AUC-ROC:   1.0000\n"
    f"AUC-PR:    1.0000\n"
    f"F1:        1.0000\n"
    f"Precision: 1.0000\n"
    f"Recall:    1.0000\n"
    f"Threshold: 0.1161"
)
ax_a.text(0.40, 0.38, _metrics_text,
          transform=ax_a.transAxes, fontsize=8.5,
          color=_TEXT_PRI, va="center", ha="left",
          bbox=dict(boxstyle="round,pad=0.5", facecolor=_PANEL_BG,
                    edgecolor=_GRID, alpha=0.95))

_lines_a, _labs_a = ax_a.get_legend_handles_labels()
_lines_a2, _labs_a2 = _ax_a2.get_legend_handles_labels()
ax_a.legend(_lines_a + _lines_a2, _labs_a + _labs_a2,
            loc="lower left", facecolor=_PANEL_BG, edgecolor=_GRID,
            labelcolor=_TEXT_PRI, fontsize=8.5)

# ── Panel B: Feature Importance ─────────────────────────────────────────────
ax_b = summary_dashboard.add_subplot(gs[0, 1])
_bars_b = ax_b.barh(_sorted_names_fi, _sorted_vals_fi,
                    color=_fi_colors, edgecolor="none", height=0.65)
for _bar_b, _val_b in zip(_bars_b, _sorted_vals_fi):
    ax_b.text(_val_b + _sorted_vals_fi.max() * 0.015,
              _bar_b.get_y() + _bar_b.get_height() / 2,
              f"{_val_b:.4f}", va="center", ha="left",
              fontsize=8, color=_TEXT_PRI)

ax_b.set_xlabel("Feature Importance (MDI)", fontsize=10)
ax_b.set_title("B  ·  Feature Importance (RF)", fontsize=13, fontweight="bold",
               color=_TEXT_PRI, loc="left", pad=10)
ax_b.set_xlim([0, _sorted_vals_fi.max() * 1.35])
ax_b.grid(axis="x", alpha=0.25)
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)

_top3_patch = mpatches.Patch(color=_YELLOW, label="Top 3 predictors")
_rest_patch  = mpatches.Patch(color=_BLUE,   label="Other features")
ax_b.legend(handles=[_top3_patch, _rest_patch],
            facecolor=_PANEL_BG, edgecolor=_GRID,
            labelcolor=_TEXT_PRI, fontsize=8.5, loc="lower right")

# ── Panel C: Segment Success Rates ──────────────────────────────────────────
ax_c = summary_dashboard.add_subplot(gs[1, 0])

_bar_w = 0.5
_x_seg = np.arange(len(_seg_names_exec))
_bars_c = ax_c.bar(_x_seg, _seg_success_exec, color=_seg_colors_exec,
                   width=_bar_w, edgecolor="none", alpha=0.90)
ax_c.axhline(0.148, color=_YELLOW, lw=1.8, linestyle="--",
             label=f"Overall baseline (0.15%)")

for _bar_c, _sr_c, _n_c in zip(_bars_c, _seg_success_exec, _seg_n_exec):
    _label_val = f"{_sr_c:.1f}%" if _sr_c > 0 else "0%"
    ax_c.text(_bar_c.get_x() + _bar_c.get_width() / 2,
              max(_bar_c.get_height() + 0.5, 1.0),
              _label_val, ha="center", va="bottom",
              fontsize=12, fontweight="bold", color=_TEXT_PRI)
    ax_c.text(_bar_c.get_x() + _bar_c.get_width() / 2,
              0.3, f"n={_n_c:,}", ha="center", va="bottom",
              fontsize=8, color=_BG if _sr_c > 1 else _TEXT_SEC, fontweight="bold")

ax_c.set_xticks(_x_seg)
ax_c.set_xticklabels(_seg_names_exec, fontsize=10, color=_TEXT_PRI)
ax_c.set_ylabel("Success Rate (%)", fontsize=10)
ax_c.set_title("C  ·  Segment Success Rates (KMeans k=3)", fontsize=13,
               fontweight="bold", color=_TEXT_PRI, loc="left", pad=10)
ax_c.set_ylim(0, max(_seg_success_exec) * 1.45 + 2)
ax_c.grid(axis="y", alpha=0.25)
ax_c.spines["top"].set_visible(False)
ax_c.spines["right"].set_visible(False)
ax_c.tick_params(axis="x", colors=_TEXT_PRI)

_baseline_line_c = plt.Line2D([0], [0], color=_YELLOW, lw=2, linestyle="--",
                               label="Baseline (0.15%)")
ax_c.legend(handles=[_baseline_line_c],
            facecolor=_PANEL_BG, edgecolor=_GRID,
            labelcolor=_TEXT_PRI, fontsize=8.5, loc="upper right")

# ── Panel D: Activation Thresholds & Lift ───────────────────────────────────
ax_d = summary_dashboard.add_subplot(gs[1, 1])

_x_thr = np.arange(len(_thr_features_exec))
_bars_d = ax_d.bar(_x_thr, _thr_sr_exec, color=_thr_colors_exec,
                   width=0.5, edgecolor="none", alpha=0.90)
ax_d.axhline(0.148, color=_YELLOW, lw=1.8, linestyle="--", alpha=0.9)

for _bar_d, _sr_d, _lift_d in zip(_bars_d, _thr_sr_exec, _thr_lift_exec):
    _sr_label = f"{_sr_d:.0f}%" if _sr_d >= 1 else f"{_sr_d:.1f}%"
    ax_d.text(_bar_d.get_x() + _bar_d.get_width() / 2,
              _bar_d.get_height() + max(_thr_sr_exec) * 0.025,
              _sr_label, ha="center", va="bottom",
              fontsize=11, fontweight="bold", color=_TEXT_PRI)
    ax_d.text(_bar_d.get_x() + _bar_d.get_width() / 2,
              max(_bar_d.get_height() * 0.45, 1.5),
              f"{_lift_d}×", ha="center", va="bottom",
              fontsize=10, color=_BG if _sr_d > 2 else _TEXT_SEC, fontweight="bold")

ax_d.set_xticks(_x_thr)
ax_d.set_xticklabels(_thr_features_exec, fontsize=9.5, color=_TEXT_PRI)
ax_d.set_ylabel("Success Rate at Threshold (%)", fontsize=10)
ax_d.set_title("D  ·  Activation Thresholds & Lift", fontsize=13,
               fontweight="bold", color=_TEXT_PRI, loc="left", pad=10)
ax_d.set_ylim(0, max(_thr_sr_exec) * 1.40 + 5)
ax_d.grid(axis="y", alpha=0.25)
ax_d.spines["top"].set_visible(False)
ax_d.spines["right"].set_visible(False)
ax_d.tick_params(axis="x", colors=_TEXT_PRI)

_baseline_line_d = plt.Line2D([0], [0], color=_YELLOW, lw=2, linestyle="--",
                               label="Baseline (0.15%)")
_lift_note = mpatches.Patch(color=_GRID, label="Numbers = lift over baseline")
ax_d.legend(handles=[_baseline_line_d, _lift_note],
            facecolor=_PANEL_BG, edgecolor=_GRID,
            labelcolor=_TEXT_PRI, fontsize=8.5, loc="upper right")

plt.savefig("executive_summary_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=_BG)
print("✅ Executive summary dashboard generated — 4 panels complete")
print(f"   Panel A: ROC + PR curves — AUC-ROC {_auc_exec:.4f}, AUC-PR {_auc_pr_exec:.4f}")
print(f"   Panel B: Feature importance — top predictor: Credits Used")
print(f"   Panel C: Segment rates — Power Users at 11.6% vs 0.15% baseline")
print(f"   Panel D: Activation thresholds — credits≥50 → 100% success (676x lift)")
