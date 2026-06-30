"""分析自迭代日志 self_iterate_log.jsonl，输出 ablation 数据与曲线。

用法:
    python analyze_self_iterate.py [日志路径]
默认读取 outputs/dashboard_demo/self_iterate_log.jsonl
产出：
    - 终端打印每个物体的逐轮 lift_delta / 成功情况
    - 按“尝试轮次”聚合的成功率（证明自迭代真的在学）
    - 若装了 matplotlib，保存一张 success_vs_attempt.png
"""
import json
import os
import sys
from collections import defaultdict


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        "outputs", "dashboard_demo", "self_iterate_log.jsonl"
    )
    if not os.path.exists(path):
        print(f"找不到日志: {path}")
        return
    records = load_records(path)
    if not records:
        print("日志为空。")
        return

    print(f"共 {len(records)} 条尝试记录\n")

    # 逐物体逐轮打印
    by_target = defaultdict(list)
    for rec in records:
        by_target[rec["target"]].append(rec)
    for target, recs in by_target.items():
        print(f"=== {target} ===")
        for rec in sorted(recs, key=lambda r: r["attempt"]):
            lift = rec.get("lift_delta")
            lift_str = f"{lift:.3f}" if isinstance(lift, (int, float)) else "N/A"
            print(
                f"  第{rec['attempt']}次  success={rec['success']}  "
                f"lift={lift_str}  drift={rec.get('obj_xy_drift')}  "
                f"state={rec.get('final_state')}  params={rec.get('params')}"
            )
        print()

    # 按尝试轮次聚合成功率
    attempts = defaultdict(lambda: [0, 0])  # attempt -> [success, total]
    for rec in records:
        bucket = attempts[rec["attempt"]]
        bucket[1] += 1
        if rec["success"]:
            bucket[0] += 1
    print("=== 成功率 vs 尝试轮次（ablation 核心数据）===")
    xs, ys = [], []
    for attempt in sorted(attempts):
        success, total = attempts[attempt]
        rate = success / total if total else 0.0
        xs.append(attempt)
        ys.append(rate)
        print(f"  第{attempt}次尝试: {success}/{total} = {rate:.0%}")

    # 累计成功率（到第 N 轮为止解决的物体占比）——更直观
    solved = set()
    cumulative = []
    targets = set(r["target"] for r in records)
    for attempt in sorted(attempts):
        for rec in records:
            if rec["attempt"] == attempt and rec["success"]:
                solved.add(rec["target"])
        cumulative.append(len(solved) / len(targets) if targets else 0.0)
    print("\n=== 累计解决率（到第 N 轮为止）===")
    for attempt, rate in zip(sorted(attempts), cumulative):
        print(f"  ≤第{attempt}轮: {rate:.0%} 的物体已成功")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, cumulative, "o-", label="Cumulative solve rate")
        ax.plot(xs, ys, "s--", alpha=0.6, label="Per-attempt success rate")
        ax.set_xlabel("Iteration attempt")
        ax.set_ylabel("Rate")
        ax.set_title("Self-iteration: grasp success vs LLM refinement rounds")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(alpha=0.3)
        out_png = os.path.join(os.path.dirname(path) or ".", "success_vs_attempt.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        print(f"\n📈 曲线已保存: {out_png}")
    except ImportError:
        print("\n(未装 matplotlib，跳过画图；pip install matplotlib 可生成曲线)")


if __name__ == "__main__":
    main()
