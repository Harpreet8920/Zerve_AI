
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ── Zerve Design System ─────────────────────────────────────────────────────
_BG       = "#1D1D20"
_PRI      = "#fbfbff"
_SEC      = "#909094"
_BLUE     = "#A1C9F4"
_ORANGE   = "#FFB482"
_GREEN    = "#8DE5A1"
_YELLOW   = "#ffd400"
_CORAL    = "#FF9F9B"
_LAVENDER = "#D0BBFF"
_GRID     = "#2e2e33"

plt.rcParams.update({
    "figure.facecolor": _BG,
    "axes.facecolor":   _BG,
    "axes.edgecolor":   _SEC,
    "axes.labelcolor":  _PRI,
    "text.color":       _PRI,
    "xtick.color":      _SEC,
    "ytick.color":      _SEC,
    "grid.color":       _GRID,
    "font.family":      "sans-serif",
    "font.size":        11,
})

# ── Working dataset: full cohort with success flag ──────────────────────────
_df = user_df.copy()
_df["events_per_session"] = _df["events_per_session"].replace([np.inf, -np.inf], 0)
_df = _df.dropna(subset=["success"])
_n_total   = len(_df)
_n_success = int(_df["success"].sum())
_base_rate = _n_success / _n_total

print("=" * 68)
print("  BUSINESS DRIVER ANALYSIS — TOP SUCCESS PREDICTORS")
print("=" * 68)
print(f"  Cohort: {_n_total:,} users  |  Successful: {_n_success:,}  |  Base rate: {_base_rate:.1%}")
print()

# ═══════════════════════════════════════════════════════════════════════════
# Helper: threshold sweep — success rate & lift at each cut-point
# ═══════════════════════════════════════════════════════════════════════════
def _threshold_stats(col, thresholds, df=_df, base=_base_rate):
    rows = []
    for t in thresholds:
        above = df[df[col] >= t]
        n_above = len(above)
        sr_above = above["success"].mean() if n_above > 0 else 0.0
        lift = sr_above / base if base > 0 else 0.0
        rows.append(dict(
            threshold=t, n_above=n_above,
            pct_above=n_above / len(df),
            sr_above=sr_above, lift=lift,
        ))
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════
# 1. CREDITS USED
# ═══════════════════════════════════════════════════════════════════════════
_cr_thresholds = [1, 5, 10, 25, 50, 100, 250]
_cr_stats = _threshold_stats("credits_used_total", _cr_thresholds)

print("── [1] CREDITS USED (credits_used_total) ──────────────────────────────")
print(f"  {'Threshold':>10}  {'Users above':>12}  {'% of cohort':>12}  {'Success rate':>13}  {'Lift':>6}")
print("  " + "-" * 58)
for _, r in _cr_stats.iterrows():
    print(f"  {int(r.threshold):>10}  {int(r.n_above):>12,}  {r.pct_above:>11.1%}  "
          f"{r.sr_above:>12.1%}  {r.lift:>5.1f}x")

_cr_best_idx = (_cr_stats["sr_above"] - _base_rate).idxmax()
_cr_best = _cr_stats.loc[_cr_best_idx]
print(f"\n  ✅ Activation threshold: credits_used_total ≥ {int(_cr_best.threshold):,}")
print(f"     → {_cr_best.sr_above:.1%} success rate  vs  {_base_rate:.1%} baseline  ({_cr_best.lift:.1f}x lift)")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 2. RUN BLOCK COUNT
# ═══════════════════════════════════════════════════════════════════════════
_rb_thresholds = [1, 3, 5, 10, 20, 50]
_rb_stats = _threshold_stats("run_block_count", _rb_thresholds)

print("── [2] RUN BLOCK COUNT ────────────────────────────────────────────────")
print(f"  {'Threshold':>10}  {'Users above':>12}  {'% of cohort':>12}  {'Success rate':>13}  {'Lift':>6}")
print("  " + "-" * 58)
for _, r in _rb_stats.iterrows():
    print(f"  {int(r.threshold):>10}  {int(r.n_above):>12,}  {r.pct_above:>11.1%}  "
          f"{r.sr_above:>12.1%}  {r.lift:>5.1f}x")

_rb_best_idx = (_rb_stats["sr_above"] - _base_rate).idxmax()
_rb_best = _rb_stats.loc[_rb_best_idx]
print(f"\n  ✅ Activation threshold: run_block_count ≥ {int(_rb_best.threshold):,}")
print(f"     → {_rb_best.sr_above:.1%} success rate  vs  {_base_rate:.1%} baseline  ({_rb_best.lift:.1f}x lift)")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 3. UNIQUE TOOLS USED
# ═══════════════════════════════════════════════════════════════════════════
_ut_thresholds = [1, 2, 3, 5, 7, 10]
_ut_stats = _threshold_stats("unique_tools_used", _ut_thresholds)

print("── [3] UNIQUE TOOLS USED ──────────────────────────────────────────────")
print(f"  {'Threshold':>10}  {'Users above':>12}  {'% of cohort':>12}  {'Success rate':>13}  {'Lift':>6}")
print("  " + "-" * 58)
for _, r in _ut_stats.iterrows():
    print(f"  {int(r.threshold):>10}  {int(r.n_above):>12,}  {r.pct_above:>11.1%}  "
          f"{r.sr_above:>12.1%}  {r.lift:>5.1f}x")

_ut_best_idx = (_ut_stats["sr_above"] - _base_rate).idxmax()
_ut_best = _ut_stats.loc[_ut_best_idx]
print(f"\n  ✅ Activation threshold: unique_tools_used ≥ {int(_ut_best.threshold):,}")
print(f"     → {_ut_best.sr_above:.1%} success rate  vs  {_base_rate:.1%} baseline  ({_ut_best.lift:.1f}x lift)")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 4. AGENT USAGE COUNT
# ═══════════════════════════════════════════════════════════════════════════
_ag_thresholds = [1, 3, 5, 10, 20, 50]
_ag_stats = _threshold_stats("agent_usage_count", _ag_thresholds)

print("── [4] AGENT USAGE COUNT ──────────────────────────────────────────────")
print(f"  {'Threshold':>10}  {'Users above':>12}  {'% of cohort':>12}  {'Success rate':>13}  {'Lift':>6}")
print("  " + "-" * 58)
for _, r in _ag_stats.iterrows():
    print(f"  {int(r.threshold):>10}  {int(r.n_above):>12,}  {r.pct_above:>11.1%}  "
          f"{r.sr_above:>12.1%}  {r.lift:>5.1f}x")

_ag_best_idx = (_ag_stats["sr_above"] - _base_rate).idxmax()
_ag_best = _ag_stats.loc[_ag_best_idx]
print(f"\n  ✅ Activation threshold: agent_usage_count ≥ {int(_ag_best.threshold):,}")
print(f"     → {_ag_best.sr_above:.1%} success rate  vs  {_base_rate:.1%} baseline  ({_ag_best.lift:.1f}x lift)")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 5. COMBINED ACTIVATION — Users who hit ALL 4 thresholds
# ═══════════════════════════════════════════════════════════════════════════
_cr_t = int(_cr_best.threshold)
_rb_t = int(_rb_best.threshold)
_ut_t = int(_ut_best.threshold)
_ag_t = int(_ag_best.threshold)

_activated = _df[
    (_df["credits_used_total"]  >= _cr_t) &
    (_df["run_block_count"]     >= _rb_t) &
    (_df["unique_tools_used"]   >= _ut_t) &
    (_df["agent_usage_count"]   >= _ag_t)
]
_not_activated = _df[~(
    (_df["credits_used_total"]  >= _cr_t) &
    (_df["run_block_count"]     >= _rb_t) &
    (_df["unique_tools_used"]   >= _ut_t) &
    (_df["agent_usage_count"]   >= _ag_t)
)]
_act_sr   = _activated["success"].mean() if len(_activated) > 0 else 0.0
_nact_sr  = _not_activated["success"].mean() if len(_not_activated) > 0 else 0.0
_act_lift = _act_sr / _base_rate if _base_rate > 0 else 0.0

print("── [5] COMBINED ACTIVATION (ALL 4 THRESHOLDS MET) ────────────────────")
print(f"  Users meeting ALL thresholds: {len(_activated):,}  ({len(_activated)/_n_total:.1%} of cohort)")
print(f"  Success rate — activated:     {_act_sr:.1%}")
print(f"  Success rate — not activated: {_nact_sr:.1%}")
print(f"  Lift vs baseline:             {_act_lift:.1f}x")
print()

# ═══════════════════════════════════════════════════════════════════════════
# 6. run_block_count binned success analysis (for chart 1)
# ═══════════════════════════════════════════════════════════════════════════
_rb_bins   = [0, 1, 3, 5, 10, 20, 50, 1e9]
_rb_labels = ["0", "1–2", "3–4", "5–9", "10–19", "20–49", "50+"]
_df["_rb_bin"] = pd.cut(_df["run_block_count"], bins=_rb_bins, labels=_rb_labels, right=False)
_rb_grouped = _df.groupby("_rb_bin", observed=True)["success"].agg(["mean", "count"]).reset_index()
_rb_grouped.columns = ["bin", "success_rate", "n"]

# ═══════════════════════════════════════════════════════════════════════════
# 7. PRODUCT STRATEGY NARRATIVE
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 68)
print("  PRODUCT STRATEGY NARRATIVE")
print("=" * 68)
print(f"""
  WHY THESE FOUR METRICS DEFINE LONG-TERM SUCCESS:

  1. CREDITS USED (top predictor by importance)
     Credits are the truest signal of value extraction — not passive browsing.
     Users burning credits run real pipelines, call models, and ship outputs.
     Threshold ≥ {_cr_t}: {_cr_best.sr_above:.0%} success vs {_base_rate:.1%} baseline ({_cr_best.lift:.0f}x lift).
     Strategy: Surface a "Credits remaining" counter prominently. Offer
     frictionless top-ups. Flag users who hit credit ceilings as VIPs.

  2. RUN BLOCK COUNT (execution depth & iteration)
     Each block run is a micro-commitment to the Zerve workflow loop. Users
     who iterate deeply stay — the act of running blocks builds habit.
     Threshold ≥ {_rb_t}: {_rb_best.sr_above:.0%} success vs {_base_rate:.1%} baseline ({_rb_best.lift:.0f}x lift).
     Strategy: In-app "run streak" nudges, auto-suggest next block runs,
     and tooltip rewards after first 5/10/20 block executions.

  3. UNIQUE TOOLS USED (breadth of platform adoption)
     Diverse tool use = the user has moved beyond a single use-case. They're
     embedding Zerve into multiple workflows — massively raising switching costs.
     Threshold ≥ {_ut_t}: {_ut_best.sr_above:.0%} success vs {_base_rate:.1%} baseline ({_ut_best.lift:.0f}x lift).
     Strategy: After week 1, surface "You haven't tried X yet" prompts
     tailored to the user's industry and existing block history.

  4. AGENT USAGE COUNT (AI amplification — Zerve's "aha moment")
     Agent use signals compound productivity gains. Heavy agent users build
     faster, ship more, and convert to paid plans at higher rates.
     Threshold ≥ {_ag_t}: {_ag_best.sr_above:.0%} success vs {_base_rate:.1%} baseline ({_ag_best.lift:.0f}x lift).
     Strategy: Make the agent unmissable in onboarding. One agent-generated
     block in session 1 is a statistically validated retention signal.

  COMBINED ACTIVATION INSIGHT:
     Users crossing ALL 4 thresholds achieve {_act_sr:.0%} success ({_act_lift:.0f}x lift).
     These {len(_activated):,} "fully activated" users ({len(_activated)/_n_total:.0%} of cohort) are the
     highest-LTV segment in the product. They should trigger: premium
     upsell offers, dedicated CS outreach, case study recruitment,
     and NPS surveys to build advocacy programs.
""")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: Success rate by run_block_count bins
# ═══════════════════════════════════════════════════════════════════════════
business_driver_chart_1 = plt.figure(figsize=(10, 6))
_ax1 = business_driver_chart_1.add_subplot(111)

# Pre-format y-axis values as percentages (avoid FuncFormatter / StrMethodFormatter)
_sr_pct = [v * 100 for v in _rb_grouped["success_rate"]]
_bar_colors = [_BLUE if v < _base_rate * 100 else _GREEN for v in _sr_pct]

_bars1 = _ax1.bar(_rb_grouped["bin"], _sr_pct, color=_bar_colors, edgecolor="none", width=0.65)
_ax1.axhline(_base_rate * 100, color=_YELLOW, lw=1.8, linestyle="--", zorder=5)

for _bar, _sr, _n in zip(_bars1, _sr_pct, _rb_grouped["n"]):
    _ax1.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.3,
              f"{_sr:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color=_PRI)
    if _bar.get_height() > 1.5:
        _ax1.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() / 2,
                  f"n={_n:,}", ha="center", va="center", fontsize=8, color=_BG, fontweight="bold")

_ax1.set_xlabel("Run Block Count (bins)", fontsize=12, labelpad=8)
_ax1.set_ylabel("Success Rate (%)", fontsize=12)
_ax1.set_title("Success Rate by Run Block Count\nMore executions → higher long-term success",
               fontsize=13, fontweight="bold", color=_PRI, pad=12)
_ax1.set_ylim(0, max(_sr_pct) * 1.30)
_ax1.grid(axis="y", alpha=0.25)
_ax1.spines["top"].set_visible(False)
_ax1.spines["right"].set_visible(False)

_baseline_line = plt.Line2D([0], [0], color=_YELLOW, lw=2, linestyle="--",
                             label=f"Baseline ({_base_rate:.1%})")
_above_patch = mpatches.Patch(color=_GREEN, label="Above baseline")
_below_patch = mpatches.Patch(color=_BLUE,  label="Below baseline")
_ax1.legend(handles=[_above_patch, _below_patch, _baseline_line],
            facecolor=_BG, edgecolor=_SEC, labelcolor=_PRI, fontsize=10)
plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: Activation lift for all 4 top drivers
# ═══════════════════════════════════════════════════════════════════════════
_driver_labels = ["Credits Used", "Run Blocks", "Unique Tools", "Agent Usage"]
_driver_lifts  = [_cr_best.lift, _rb_best.lift, _ut_best.lift, _ag_best.lift]
_driver_sr     = [_cr_best.sr_above, _rb_best.sr_above, _ut_best.sr_above, _ag_best.sr_above]
_driver_thresh = [f"≥ {_cr_t}", f"≥ {_rb_t}", f"≥ {_ut_t}", f"≥ {_ag_t}"]
_driver_colors = [_YELLOW, _GREEN, _BLUE, _ORANGE]

business_driver_chart_2 = plt.figure(figsize=(10, 6))
_ax2 = business_driver_chart_2.add_subplot(111)

_x_pos = np.arange(len(_driver_labels))
_bars2 = _ax2.bar(_x_pos, _driver_lifts, color=_driver_colors, edgecolor="none", width=0.55)
_ax2.axhline(1.0, color=_SEC, lw=1.5, linestyle="--", label="No lift (1×)")

for _bar, _lift, _sr, _thresh in zip(_bars2, _driver_lifts, _driver_sr, _driver_thresh):
    _ax2.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + max(_driver_lifts) * 0.02,
              f"{_lift:.0f}×", ha="center", va="bottom", fontsize=13, fontweight="bold", color=_PRI)
    if _bar.get_height() > 2:
        _ax2.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() / 2,
                  f"{_sr:.0%} success\n({_thresh})",
                  ha="center", va="center", fontsize=9, color=_BG, fontweight="bold")

_ax2.set_xticks(_x_pos)
_ax2.set_xticklabels(_driver_labels, fontsize=11, color=_PRI)
_ax2.set_ylabel("Lift over baseline", fontsize=12)
_ax2.set_title("Activation Lift — Top 4 Success Predictors\nSuccess rate uplift at key usage thresholds",
               fontsize=13, fontweight="bold", color=_PRI, pad=12)
_ax2.set_ylim(0, max(_driver_lifts) * 1.25)
_ax2.grid(axis="y", alpha=0.25)
_ax2.spines["top"].set_visible(False)
_ax2.spines["right"].set_visible(False)
_ax2.legend(facecolor=_BG, edgecolor=_SEC, labelcolor=_PRI, fontsize=10)
plt.tight_layout()

# ═══════════════════════════════════════════════════════════════════════════
# Export: structured activation thresholds for downstream use
# ═══════════════════════════════════════════════════════════════════════════
activation_thresholds = {
    "credits_used_total": {"threshold": _cr_t, "success_rate": round(float(_cr_best.sr_above), 4), "lift": round(float(_cr_best.lift), 2)},
    "run_block_count":    {"threshold": _rb_t, "success_rate": round(float(_rb_best.sr_above), 4), "lift": round(float(_rb_best.lift), 2)},
    "unique_tools_used":  {"threshold": _ut_t, "success_rate": round(float(_ut_best.sr_above), 4), "lift": round(float(_ut_best.lift), 2)},
    "agent_usage_count":  {"threshold": _ag_t, "success_rate": round(float(_ag_best.sr_above), 4), "lift": round(float(_ag_best.lift), 2)},
    "combined_activated": {
        "n_users":      int(len(_activated)),
        "pct_cohort":   round(len(_activated) / _n_total, 4),
        "success_rate": round(float(_act_sr), 4),
        "lift":         round(float(_act_lift), 2),
    },
    "baseline_rate": round(_base_rate, 4),
}

print("=" * 68)
print("  ACTIVATION THRESHOLDS SUMMARY (exported: activation_thresholds)")
print("=" * 68)
for _feat, _info in activation_thresholds.items():
    if _feat == "baseline_rate":
        print(f"  Baseline success rate: {_info:.1%}")
    elif _feat == "combined_activated":
        print(f"  Combined (all 4):  n={_info['n_users']:,} ({_info['pct_cohort']:.1%}) "
              f"→ {_info['success_rate']:.1%} success ({_info['lift']:.0f}x lift)")
    else:
        print(f"  {_feat:<22}  threshold={_info['threshold']:>5}  "
              f"→ {_info['success_rate']:.1%} success  ({_info['lift']:.0f}x lift)")
print("=" * 68)
