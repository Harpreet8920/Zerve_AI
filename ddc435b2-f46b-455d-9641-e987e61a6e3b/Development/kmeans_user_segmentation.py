import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Design system ──────────────────────────────────────────────────────────────
BG_SEG       = "#1D1D20"
TEXT_PRI_SEG = "#fbfbff"
TEXT_SEC_SEG = "#909094"
GRID_SEG     = "#2e2e34"
GREEN_SEG    = "#8DE5A1"
BLUE_SEG     = "#A1C9F4"
ORANGE_SEG   = "#FFB482"

SEGMENT_COLORS = [GREEN_SEG, BLUE_SEG, ORANGE_SEG]

# ── Feature set for clustering (most behaviorally discriminative) ──────────────
CLUSTER_FEATURES = [
    "total_events", "active_days", "run_block_count",
    "agent_usage_count", "blocks_created", "credits_used_total",
    "unique_tools_used", "events_per_session"
]

cluster_input = X[CLUSTER_FEATURES].copy()
cluster_input = cluster_input.replace([np.inf, -np.inf], 0).fillna(0)

# ── Log1p transform → tames power-law tails for meaningful k-means clusters ───
cluster_log = np.log1p(cluster_input)

# ── StandardScale ──────────────────────────────────────────────────────────────
seg_scaler = StandardScaler()
X_scaled_seg = seg_scaler.fit_transform(cluster_log)

# ── KMeans k=3  (n_init=30 for stability) ─────────────────────────────────────
kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=30, max_iter=600)
segment_labels = kmeans_model.fit_predict(X_scaled_seg)

# ── Enrich dataframe with raw metrics + success label ─────────────────────────
seg_df = cluster_input.copy()
seg_df["segment"] = segment_labels
seg_df["success"] = y.values

# ── Per-segment statistics (on RAW values for interpretability) ─────────────────
seg_profile = seg_df.groupby("segment").agg(
    n_users            = ("success",           "count"),
    success_rate       = ("success",           "mean"),
    avg_events         = ("total_events",       "mean"),
    avg_active_days    = ("active_days",        "mean"),
    avg_run_blocks     = ("run_block_count",    "mean"),
    avg_agent_usage    = ("agent_usage_count",  "mean"),
    avg_blocks_created = ("blocks_created",     "mean"),
    avg_credits        = ("credits_used_total", "mean"),
    avg_tools          = ("unique_tools_used",  "mean"),
    avg_eps            = ("events_per_session", "mean"),
).reset_index()

# ── Auto-label: Power Users → highest success_rate; At-Risk → lowest ─────────
sorted_idx = seg_profile["success_rate"].argsort()[::-1].values  # high → low
archetype_names  = ["Power Users", "Casual Explorers", "At-Risk Users"]
archetype_emojis = ["🚀", "🔍", "⚠️"]

label_map   = {}
emoji_map   = {}
success_map = {}
for rank, seg_id in enumerate(sorted_idx):
    seg_int = seg_profile.loc[seg_id, "segment"]
    label_map[seg_int]   = archetype_names[rank]
    emoji_map[seg_int]   = archetype_emojis[rank]
    success_map[seg_int] = seg_profile.loc[seg_id, "success_rate"]

seg_profile["archetype"] = seg_profile["segment"].map(label_map)
seg_profile["emoji"]     = seg_profile["segment"].map(emoji_map)

# ── Print summary ──────────────────────────────────────────────────────────────
print("=" * 74)
print(f"{'USER BEHAVIORAL SEGMENTATION  ·  KMeans k=3 (log-scaled)':^74}")
print("=" * 74)
print(f"{'Segment':<22} {'Users':>6} {'Success%':>10} {'Avg Credits':>13} "
      f"{'Avg Blocks':>11} {'Avg Agent':>10}")
print("-" * 74)
for _, row in seg_profile.sort_values("success_rate", ascending=False).iterrows():
    print(f"{row['emoji']} {row['archetype']:<20} {row['n_users']:>6,} "
          f"{row['success_rate']*100:>9.1f}%  {row['avg_credits']:>12.1f} "
          f"{row['avg_blocks_created']:>11.1f} {row['avg_agent_usage']:>10.1f}")
print("=" * 74)

# ── ① Radar / Spider chart ─────────────────────────────────────────────────────
radar_metrics = {
    "Active Days":    "avg_active_days",
    "Run Blocks":     "avg_run_blocks",
    "Agent Usage":    "avg_agent_usage",
    "Blocks Created": "avg_blocks_created",
    "Credits Used":   "avg_credits",
    "Tools Used":     "avg_tools",
    "Events/Session": "avg_eps",
}
radar_labels = list(radar_metrics.keys())
radar_cols   = list(radar_metrics.values())
N = len(radar_labels)

raw_vals  = seg_profile[radar_cols].values.astype(float)
col_max   = raw_vals.max(axis=0)
col_max   = np.where(col_max == 0, 1, col_max)
norm_vals = raw_vals / col_max

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

radar_chart = plt.figure(figsize=(9, 9))
radar_chart.patch.set_facecolor(BG_SEG)
ax_r = radar_chart.add_subplot(111, polar=True)
ax_r.set_facecolor(BG_SEG)

for level in [0.25, 0.5, 0.75, 1.0]:
    ring = [level] * N + [level]
    ax_r.plot(angles, ring, color=GRID_SEG, linewidth=0.7, linestyle="--")
    ax_r.fill(angles, ring, color=GRID_SEG, alpha=0.04)

for i, row in seg_profile.iterrows():
    rank  = list(sorted_idx).index(row["segment"])
    color = SEGMENT_COLORS[rank]
    vals  = norm_vals[i].tolist() + [norm_vals[i][0]]
    ax_r.plot(angles, vals, color=color, linewidth=2.5)
    ax_r.fill(angles, vals, color=color, alpha=0.15)

ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(radar_labels, color=TEXT_PRI_SEG, fontsize=11, fontweight="bold")
ax_r.set_yticks([])
ax_r.spines["polar"].set_color(GRID_SEG)
ax_r.grid(color=GRID_SEG, linewidth=0.5)

legend_patches = []
for rank, seg_id in enumerate(sorted_idx):
    row = seg_profile[seg_profile["segment"] == seg_id].iloc[0]
    lbl = (f"{row['emoji']} {row['archetype']}  "
           f"(n={row['n_users']:,}  ·  {row['success_rate']*100:.1f}% success)")
    legend_patches.append(mpatches.Patch(color=SEGMENT_COLORS[rank], label=lbl))

ax_r.legend(
    handles=legend_patches, loc="upper right",
    bbox_to_anchor=(1.44, 1.18), frameon=True,
    facecolor="#2a2a2e", edgecolor=GRID_SEG, fontsize=10, labelcolor=TEXT_PRI_SEG,
)
ax_r.set_title(
    "User Behavioral Archetypes  ·  Segment Radar Profiles",
    color=TEXT_PRI_SEG, fontsize=14, fontweight="bold", pad=28, y=1.08,
)
sub = "  |  ".join(
    f"{row['emoji']} {row['archetype']}: {row['success_rate']*100:.1f}% success"
    for _, row in seg_profile.sort_values("success_rate", ascending=False).iterrows()
)
ax_r.annotate(sub, xy=(0.5, -0.06), xycoords="axes fraction",
              ha="center", fontsize=9, color=TEXT_SEC_SEG)
plt.tight_layout()

# ── ② Grouped bar chart ────────────────────────────────────────────────────────
bar_metrics  = ["avg_run_blocks", "avg_agent_usage", "avg_blocks_created",
                "avg_credits",    "avg_tools",        "avg_active_days"]
bar_labels_x = ["Run Blocks", "Agent Usage", "Blocks Created",
                "Credits Used", "Tools Used", "Active Days"]

bar_chart = plt.figure(figsize=(13, 6))
bar_chart.patch.set_facecolor(BG_SEG)
ax_b = bar_chart.add_subplot(111)
ax_b.set_facecolor(BG_SEG)

width = 0.25
x_pos = np.arange(len(bar_metrics))

for rank, seg_id in enumerate(sorted_idx):
    row    = seg_profile[seg_profile["segment"] == seg_id].iloc[0]
    vals   = [row[m] for m in bar_metrics]
    offset = (rank - 1) * width
    ax_b.bar(
        x_pos + offset, vals, width=width - 0.02,
        color=SEGMENT_COLORS[rank], alpha=0.88,
        label=f"{row['emoji']} {row['archetype']} ({row['success_rate']*100:.1f}% success)",
    )

ax_b.set_xticks(x_pos)
ax_b.set_xticklabels(bar_labels_x, color=TEXT_PRI_SEG, fontsize=11)
ax_b.tick_params(axis="y", colors=TEXT_SEC_SEG)
ax_b.set_ylabel("Mean Value (raw units)", color=TEXT_SEC_SEG, fontsize=10)
ax_b.set_title(
    "Segment Behavioral Profiles  ·  Grouped Bar Comparison",
    color=TEXT_PRI_SEG, fontsize=14, fontweight="bold", pad=16,
)
ax_b.spines[["top", "right", "left", "bottom"]].set_color(GRID_SEG)
ax_b.tick_params(axis="x", colors=TEXT_PRI_SEG)
ax_b.yaxis.grid(True, color=GRID_SEG, linewidth=0.5, linestyle="--")
ax_b.set_axisbelow(True)
ax_b.legend(frameon=True, facecolor="#2a2a2e", edgecolor=GRID_SEG,
            fontsize=10, labelcolor=TEXT_PRI_SEG, loc="upper right")
plt.tight_layout()
plt.show()

# ── Export artefacts ───────────────────────────────────────────────────────────
segment_profile_df  = seg_profile
segment_assignments = segment_labels
segment_label_map   = label_map
segment_success_map = success_map
kmeans_seg_model    = kmeans_model
