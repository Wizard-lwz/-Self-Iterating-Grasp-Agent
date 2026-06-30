"""对比启发式微调 vs LLM 调参的学习效率。

用法:
    python compare_heuristic_vs_llm.py \
        outputs/dashboard_demo/self_iterate_log.jsonl \
        outputs/llm_iterate/self_iterate_log.jsonl

产出：
    - 两版的成功率曲线对比
    - 各物体的参数收敛速度对比
    - 解决相同物体所需轮次对比
"""
import json
import sys
from collections import defaultdict


def load_records(path):
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"⚠️ 找不到: {path}")
    return records


def compute_metrics(records):
    """计算关键指标：成功率 vs 轮次、首次成功轮次"""
    by_attempt = defaultdict(lambda: [0, 0])  # attempt -> [success, total]
    first_success = {}  # target -> first successful attempt

    for rec in records:
        bucket = by_attempt[rec["attempt"]]
        bucket[1] += 1
        if rec["success"]:
            bucket[0] += 1
            target = rec["target"]
            if target not in first_success:
                first_success[target] = rec["attempt"]

    success_rates = {att: (s/t if t else 0) for att, (s, t) in by_attempt.items()}

    # 累计解决率
    targets = set(r["target"] for r in records)
    solved = set()
    cumulative = {}
    for attempt in sorted(by_attempt.keys()):
        for rec in records:
            if rec["attempt"] == attempt and rec["success"]:
                solved.add(rec["target"])
        cumulative[attempt] = len(solved) / len(targets) if targets else 0

    return {
        "success_rates": success_rates,
        "cumulative": cumulative,
        "first_success": first_success,
        "total_attempts": sum(bucket[1] for bucket in by_attempt.values()),
    }


def main():
    if len(sys.argv) < 3:
        print("用法: python compare_heuristic_vs_llm.py <heuristic.jsonl> <llm.jsonl>")
        return

    heur_path, llm_path = sys.argv[1], sys.argv[2]

    heur_recs = load_records(heur_path)
    llm_recs = load_records(llm_path)

    if not heur_recs and not llm_recs:
        print("两个日志都为空,无法对比。")
        return

    print("=" * 60)
    print("启发式微调 vs LLM 调参对比")
    print("=" * 60)

    heur_m = compute_metrics(heur_recs) if heur_recs else None
    llm_m = compute_metrics(llm_recs) if llm_recs else None

    if heur_m:
        print(f"\n【启发式微调】{len(heur_recs)} 条记录")
        print("  成功率 vs 轮次:")
        for att in sorted(heur_m["success_rates"].keys()):
            print(f"    第{att}轮: {heur_m['success_rates'][att]:.0%}")
        print("  累计解决率:")
        for att in sorted(heur_m["cumulative"].keys()):
            print(f"    ≤第{att}轮: {heur_m['cumulative'][att]:.0%}")
        print("  首次成功轮次:", heur_m["first_success"] or "无")

    if llm_m:
        print(f"\n【LLM 调参】{len(llm_recs)} 条记录")
        print("  成功率 vs 轮次:")
        for att in sorted(llm_m["success_rates"].keys()):
            print(f"    第{att}轮: {llm_m['success_rates'][att]:.0%}")
        print("  累计解决率:")
        for att in sorted(llm_m["cumulative"].keys()):
            print(f"    ≤第{att}轮: {llm_m['cumulative'][att]:.0%}")
        print("  首次成功轮次:", llm_m["first_success"] or "无")

    # 对比结论
    if heur_m and llm_m:
        print("\n" + "=" * 60)
        print("对比结论")
        print("=" * 60)

        # 哪个更快解决问题
        common_targets = set(heur_m["first_success"].keys()) & set(llm_m["first_success"].keys())
        if common_targets:
            print(f"  共同解决的物体: {common_targets}")
            for t in common_targets:
                h_att = heur_m["first_success"][t]
                l_att = llm_m["first_success"][t]
                winner = "LLM" if l_att < h_att else ("启发式" if h_att < l_att else "平局")
                print(f"    {t}: 启发式第{h_att}轮 vs LLM第{l_att}轮 → {winner} 更快")

        # 最终累计解决率
        max_att = max(max(heur_m["cumulative"].keys()), max(llm_m["cumulative"].keys()))
        h_final = heur_m["cumulative"].get(max_att, 0)
        l_final = llm_m["cumulative"].get(max_att, 0)
        print(f"\n  最终累计解决率(第{max_att}轮):")
        print(f"    启发式: {h_final:.0%}")
        print(f"    LLM: {l_final:.0%}")
        if l_final > h_final:
            print("    → LLM 解决了更多物体")
        elif h_final > l_final:
            print("    → 启发式解决了更多物体")
        else:
            print("    → 平局")

    # 可视化(如果有 matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        if heur_m:
            xs = sorted(heur_m["cumulative"].keys())
            ys = [heur_m["cumulative"][x] for x in xs]
            ax.plot(xs, ys, "o-", label="Heuristic", linewidth=2)

        if llm_m:
            xs = sorted(llm_m["cumulative"].keys())
            ys = [llm_m["cumulative"][x] for x in xs]
            ax.plot(xs, ys, "s-", label="LLM", linewidth=2)

        ax.set_xlabel("Iteration attempt", fontsize=12)
        ax.set_ylabel("Cumulative solve rate", fontsize=12)
        ax.set_title("Heuristic vs LLM: learning efficiency", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        out = "heuristic_vs_llm_comparison.png"
        fig.savefig(out, dpi=120)
        print(f"\n📈 对比曲线已保存: {out}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
