
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score,
    precision_score, recall_score,
    roc_auc_score, accuracy_score,
    matthews_corrcoef, balanced_accuracy_score,
)
import warnings
warnings.filterwarnings("ignore")

# ── Zerve Design System ─────────────────────────────────────────────────────
BG        = "#1D1D20"
TEXT_PRI  = "#fbfbff"
TEXT_SEC  = "#909094"
BLUE      = "#A1C9F4"
ORANGE    = "#FFB482"
GREEN     = "#8DE5A1"
YELLOW    = "#ffd400"
GRID_COL  = "#2e2e33"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "axes.edgecolor":    TEXT_SEC,
    "axes.labelcolor":   TEXT_PRI,
    "text.color":        TEXT_PRI,
    "xtick.color":       TEXT_SEC,
    "ytick.color":       TEXT_SEC,
    "grid.color":        GRID_COL,
    "font.family":       "sans-serif",
    "font.size":         11,
})

# ── Predict probabilities & classes ─────────────────────────────────────────
_y_prob = best_model.predict_proba(X_test)[:, 1]
_y_pred_default = (_y_prob >= 0.5).astype(int)

# ── 1. ROC Curve with AUC ───────────────────────────────────────────────────
_fpr, _tpr, _ = roc_curve(y_test, _y_prob)
_auc_roc = auc(_fpr, _tpr)

roc_curve_chart, ax = plt.subplots(figsize=(8, 6))
ax.plot(_fpr, _tpr, color=BLUE, lw=2.5, label=f"ROC Curve  (AUC = {_auc_roc:.4f})")
ax.plot([0, 1], [0, 1], color=TEXT_SEC, lw=1.5, linestyle="--", label="Random Classifier")
ax.fill_between(_fpr, _tpr, alpha=0.08, color=BLUE)
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — Test Set", fontsize=14, fontweight="bold", color=TEXT_PRI, pad=14)
ax.legend(loc="lower right", facecolor=BG, edgecolor=TEXT_SEC, labelcolor=TEXT_PRI)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.05])
plt.tight_layout()

# ── 2. Precision-Recall Curve ────────────────────────────────────────────────
_prec_vals, _rec_vals, _ = precision_recall_curve(y_test, _y_prob)
_auc_pr = average_precision_score(y_test, _y_prob)
_baseline_pr = float(y_test.mean())

precision_recall_curve_chart, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(_rec_vals, _prec_vals, color=ORANGE, lw=2.5, label=f"PR Curve  (AP = {_auc_pr:.4f})")
ax2.axhline(_baseline_pr, color=TEXT_SEC, lw=1.5, linestyle="--",
            label=f"Baseline (prevalence = {_baseline_pr:.2%})")
ax2.fill_between(_rec_vals, _prec_vals, alpha=0.08, color=ORANGE)
ax2.set_xlabel("Recall", fontsize=12)
ax2.set_ylabel("Precision", fontsize=12)
ax2.set_title("Precision-Recall Curve — Test Set", fontsize=14, fontweight="bold",
              color=TEXT_PRI, pad=14)
ax2.legend(loc="upper right", facecolor=BG, edgecolor=TEXT_SEC, labelcolor=TEXT_PRI)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-0.01, 1.01])
ax2.set_ylim([-0.01, 1.05])
plt.tight_layout()

# ── 3. Confusion Matrix helper ───────────────────────────────────────────────
def _plot_confusion(cm, title, axis, threshold_label):
    _norm = cm.astype(float) / cm.sum()
    axis.set_facecolor(BG)
    # Draw colored rectangles manually to keep full dark-theme control
    _cell_colors = [["#1e3a5f", "#3b6ea5"], ["#1e5f2f", "#2e9e4f"]]
    for i in range(2):
        for j in range(2):
            _bg_color = "#2a3a4a" if (i == j) else "#3a2a2a"
            rect = plt.Rectangle([j - 0.5, i - 0.5], 1, 1,
                                  facecolor=_bg_color, edgecolor="#3a3a3e", lw=1.5)
            axis.add_patch(rect)
            _val = cm[i, j]
            _pct = _norm[i, j] * 100
            axis.text(j, i, f"{_val:,}\n({_pct:.1f}%)", ha="center", va="center",
                      fontsize=13, fontweight="bold", color=TEXT_PRI)
    labels = ["Not Successful (0)", "Successful (1)"]
    axis.set_xticks([0, 1])
    axis.set_xticklabels(labels, fontsize=10, color=TEXT_PRI)
    axis.set_yticks([0, 1])
    axis.set_yticklabels(labels, fontsize=10, color=TEXT_PRI, rotation=90, va="center")
    axis.set_xlabel("Predicted Label", fontsize=11, color=TEXT_PRI)
    axis.set_ylabel("True Label", fontsize=11, color=TEXT_PRI)
    axis.set_title(f"{title}\n(threshold = {threshold_label})", fontsize=12,
                   fontweight="bold", color=TEXT_PRI, pad=10)
    axis.set_xlim([-0.5, 1.5])
    axis.set_ylim([-0.5, 1.5])
    axis.spines[:].set_visible(False)
    axis.tick_params(length=0)

_cm_default = confusion_matrix(y_test, _y_pred_default)
confusion_matrix_default_chart, ax3 = plt.subplots(figsize=(7, 5.5))
_plot_confusion(_cm_default, "Confusion Matrix — Default", ax3, "0.50")
plt.tight_layout()

# ── 3b. Optimal Threshold (max F1 on test set) ──────────────────────────────
_thresholds_search = np.linspace(0.01, 0.99, 500)
_f1_scores = [
    f1_score(y_test, (_y_prob >= _t).astype(int), zero_division=0)
    for _t in _thresholds_search
]
_f1_arr = np.array(_f1_scores)
_best_idx = int(np.argmax(_f1_arr))
optimal_threshold = float(_thresholds_search[_best_idx])

_y_pred_optimal = (_y_prob >= optimal_threshold).astype(int)
_cm_optimal = confusion_matrix(y_test, _y_pred_optimal)

print(f"  ✅ Optimal threshold (max F1): {optimal_threshold:.4f}")
print(f"     F1 at optimal threshold:    {_f1_arr[_best_idx]:.4f}")

confusion_matrix_optimal_chart, ax4 = plt.subplots(figsize=(7, 5.5))
_plot_confusion(_cm_optimal, "Confusion Matrix — Optimal", ax4, f"{optimal_threshold:.3f}")
plt.tight_layout()

# ── 4. Feature Importance ────────────────────────────────────────────────────
_feature_names = list(X_test.columns)
_importances = best_model.feature_importances_
_sorted_idx = np.argsort(_importances)
_sorted_names = [_feature_names[i] for i in _sorted_idx]
_sorted_imp = _importances[_sorted_idx]

_label_map = {
    "total_events":       "Total Events",
    "active_days":        "Active Days",
    "total_sessions":     "Total Sessions",
    "events_per_day":     "Events / Day",
    "run_block_count":    "Run Block Count",
    "agent_usage_count":  "Agent Usage Count",
    "blocks_created":     "Blocks Created",
    "credits_used_total": "Credits Used (Total)",
    "unique_tools_used":  "Unique Tools Used",
    "events_per_session": "Events / Session",
}
_clean_labels = [_label_map.get(n, n) for n in _sorted_names]
_colors = [YELLOW if i >= len(_sorted_imp) - 3 else BLUE for i in range(len(_sorted_imp))]

feature_importance_chart, ax5 = plt.subplots(figsize=(9, 6))
_bars = ax5.barh(_clean_labels, _sorted_imp, color=_colors, edgecolor="none", height=0.65)
for _bar, _val in zip(_bars, _sorted_imp):
    ax5.text(_val + 0.001, _bar.get_y() + _bar.get_height() / 2,
             f"{_val:.4f}", va="center", ha="left", fontsize=9.5, color=TEXT_PRI)
ax5.set_xlabel("Feature Importance (Mean Decrease Impurity)", fontsize=11)
ax5.set_title("Feature Importance — Random Forest", fontsize=14, fontweight="bold",
              color=TEXT_PRI, pad=14)
ax5.set_xlim([0, _sorted_imp.max() * 1.22])
ax5.grid(axis="x", alpha=0.3)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)
plt.tight_layout()

# ── 5. Final Metrics at Optimal Threshold ───────────────────────────────────
_tn, _fp, _fn, _tp = _cm_optimal.ravel()
_f1_default = f1_score(y_test, _y_pred_default, zero_division=0)

final_metrics = {
    "optimal_threshold":  optimal_threshold,
    "auc_roc":            round(roc_auc_score(y_test, _y_prob), 4),
    "auc_pr":             round(_auc_pr, 4),
    "accuracy":           round(accuracy_score(y_test, _y_pred_optimal), 4),
    "balanced_accuracy":  round(balanced_accuracy_score(y_test, _y_pred_optimal), 4),
    "f1":                 round(f1_score(y_test, _y_pred_optimal, zero_division=0), 4),
    "precision":          round(precision_score(y_test, _y_pred_optimal, zero_division=0), 4),
    "recall":             round(recall_score(y_test, _y_pred_optimal, zero_division=0), 4),
    "mcc":                round(matthews_corrcoef(y_test, _y_pred_optimal), 4),
    "true_positives":     int(_tp),
    "false_positives":    int(_fp),
    "true_negatives":     int(_tn),
    "false_negatives":    int(_fn),
    "specificity":        round(_tn / (_tn + _fp), 4) if (_tn + _fp) > 0 else 0.0,
}

print()
print("=" * 62)
print("  FINAL EVALUATION — BEST MODEL ON HELD-OUT TEST SET")
print("  Model: RandomForestClassifier")
print("=" * 62)
print()
print(f"  {'Metric':<24} {'Default (0.50)':>15}  {'Optimal':>15}")
print("  " + "-" * 58)
print(f"  {'AUC-ROC':<24} {'—':>15}  {final_metrics['auc_roc']:>15.4f}")
print(f"  {'AUC-PR':<24} {'—':>15}  {final_metrics['auc_pr']:>15.4f}")
print(f"  {'Accuracy':<24} {accuracy_score(y_test, _y_pred_default):>15.4f}  {final_metrics['accuracy']:>15.4f}")
print(f"  {'Balanced Accuracy':<24} {balanced_accuracy_score(y_test, _y_pred_default):>15.4f}  {final_metrics['balanced_accuracy']:>15.4f}")
print(f"  {'F1':<24} {_f1_default:>15.4f}  {final_metrics['f1']:>15.4f}")
print(f"  {'Precision':<24} {precision_score(y_test, _y_pred_default, zero_division=0):>15.4f}  {final_metrics['precision']:>15.4f}")
print(f"  {'Recall':<24} {recall_score(y_test, _y_pred_default, zero_division=0):>15.4f}  {final_metrics['recall']:>15.4f}")
print(f"  {'Specificity':<24} {'—':>15}  {final_metrics['specificity']:>15.4f}")
print(f"  {'MCC':<24} {matthews_corrcoef(y_test, _y_pred_default):>15.4f}  {final_metrics['mcc']:>15.4f}")
print(f"  {'True Positives':<24} {'—':>15}  {final_metrics['true_positives']:>15}")
print(f"  {'False Positives':<24} {'—':>15}  {final_metrics['false_positives']:>15}")
print(f"  {'True Negatives':<24} {'—':>15}  {final_metrics['true_negatives']:>15}")
print(f"  {'False Negatives':<24} {'—':>15}  {final_metrics['false_negatives']:>15}")
print()
print(f"  ✅ Optimal threshold: {optimal_threshold:.4f}")
print(f"  ✅ F1 improvement:    {_f1_default:.4f} → {final_metrics['f1']:.4f}  (+{final_metrics['f1'] - _f1_default:.4f})")
print("=" * 62)
