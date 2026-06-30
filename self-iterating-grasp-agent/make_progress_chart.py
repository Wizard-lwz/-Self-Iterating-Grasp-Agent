"""生成四版进展对比图（用于 README 门面）。"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

VERSIONS = [
    ("outputs/v1_baseline_log.jsonl", "V1 Baseline", "#888888"),
    ("outputs/v2_llm_log.jsonl", "V2 LLM-tuning", "#e74c3c"),
    ("outputs/v3_improved_log.jsonl", "V3 Improved heuristic", "#3498db"),
    ("outputs/v4_optimized_log.jsonl", "V4 Shape-aware init", "#2ecc71"),
]


def cumulative_solve(path):
    try:
        recs = [json.loads(l) for l in open(path) if l.strip()]
    except FileNotFoundError:
        return [], []
    targets = set(r["target"] for r in recs)
    by_attempt = defaultdict(list)
    for r in recs:
        by_attempt[r["attempt"]].append(r)
    solved = set()
    xs, ys = [], []
    for att in sorted(by_attempt):
        for r in by_attempt[att]:
            if r["success"]:
                solved.add(r["target"])
        xs.append(att)
        ys.append(len(solved) / len(targets) if targets else 0)
    return xs, ys


fig, ax = plt.subplots(figsize=(9, 5.5))
for path, label, color in VERSIONS:
    xs, ys = cumulative_solve(path)
    if xs:
        final = ys[-1]
        ax.plot(xs, [y * 100 for y in ys], "o-", color=color, linewidth=2.5,
                markersize=7, label=f"{label}  (final {final*100:.0f}%)")

ax.set_xlabel("Iteration attempt", fontsize=13)
ax.set_ylabel("Cumulative solve rate (%)", fontsize=13)
ax.set_title("Self-Iterating Grasp Agent: progress across versions",
             fontsize=15, fontweight="bold")
ax.set_ylim(-3, 100)
ax.set_xticks(range(1, 9))
ax.legend(fontsize=11, loc="upper left", framealpha=0.95)
ax.grid(alpha=0.3)
ax.axhline(75, color="#2ecc71", linestyle=":", alpha=0.4)
fig.tight_layout()
fig.savefig("assets/progress_comparison.png", dpi=130)
print("✅ 已生成 assets/progress_comparison.png")
