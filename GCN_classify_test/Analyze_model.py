# analyze_worst_class.py
import argparse, csv
from pathlib import Path
from collections import defaultdict

def per_class_metrics(y_true, y_pred, num_classes=None):
    if num_classes is None:
        num_classes = max(max(y_true, default=0), max(y_pred, default=0)) + 1

    # 统计 TP/FP/FN、支持度（该类真实样本数）
    TP = [0]*num_classes
    FP = [0]*num_classes
    FN = [0]*num_classes
    support = [0]*num_classes
    correct = [0]*num_classes

    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if t == p:
            TP[t] += 1
            correct[t] += 1
        else:
            FP[p] += 1
            FN[t] += 1

    # 计算 per-class 指标
    rows = []
    for c in range(num_classes):
        tp, fp, fn = TP[c], FP[c], FN[c]
        sup = support[c]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        acc_c = correct[c] / sup if sup > 0 else float("nan")
        rows.append({
            "class": c,
            "support": sup,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc_c
        })

    # 宏平均/加权平均
    valid = [r for r in rows if r["support"] > 0]
    if valid:
        macro_f1 = sum(r["f1"] for r in valid) / len(valid)
        weighted_f1 = (sum(r["f1"] * r["support"] for r in valid) /
                       sum(r["support"] for r in valid))
    else:
        macro_f1 = weighted_f1 = float("nan")

    # 总体准确率
    total_correct = sum(TP)
    total = len(y_true)
    overall_acc = total_correct / total if total > 0 else float("nan")

    return rows, {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "overall_acc": overall_acc}

def main():
    ap = argparse.ArgumentParser(description="Aggregate pred_*.csv and report worst class by F1.")
    ap.add_argument("--out_dir", default="./runs/gcn_transformer_inductive",
                    help="Directory that contains pred_*.csv exported by training script.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    csv_files = sorted(out_dir.glob("pred_*.csv"))
    if not csv_files:
        raise SystemExit(f"No pred_*.csv found under: {out_dir}. "
                         "Make sure you ran training and it saved per-graph predictions.")

    y_true, y_pred = [], []
    per_graph = defaultdict(lambda: {"y_true": [], "y_pred": []})

    for f in csv_files:
        with f.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    t = int(row["y_true"])
                    p = int(row["y_pred"])
                except Exception:
                    # 跳过坏行
                    continue
                y_true.append(t)
                y_pred.append(p)
                per_graph[f.stem]["y_true"].append(t)
                per_graph[f.stem]["y_pred"].append(p)

    rows, agg = per_class_metrics(y_true, y_pred)

    # 找到最差类别（按 F1 升序，遇到并列再按支持度降序优先显示样本更多的）
    rows_sorted = sorted(rows, key=lambda r: (r["f1"], -r["support"]))
    worst = rows_sorted[0] if rows_sorted else None

    # 打印总体
    print("=== Overall Performance ===")
    print(f"Files used: {len(csv_files)}")
    print(f"Samples: {len(y_true)}")
    print(f"Overall Accuracy: {agg['overall_acc']:.4f}")
    print(f"Macro F1: {agg['macro_f1']:.4f} | Weighted F1: {agg['weighted_f1']:.4f}")
    print()

    # 打印每类（按 F1 升序）
    print("=== Per-Class Metrics (sorted by F1 asc) ===")
    header = f"{'class':>7}  {'support':>8}  {'precision':>9}  {'recall':>7}  {'f1':>7}  {'accuracy':>9}"
    print(header)
    print("-"*len(header))
    for r in rows_sorted:
        mark = "  <= worst" if worst and r["class"] == worst["class"] else ""
        acc_str = f"{r['accuracy']:.4f}" if r["support"] > 0 else "   n/a "
        print(f"{r['class']:>7}  {r['support']:>8}  {r['precision']:>9.4f}  {r['recall']:>7.4f}  {r['f1']:>7.4f}  {acc_str:>9}{mark}")
    print()

    if worst:
        print(f"*** Worst class by F1: {worst['class']} "
              f"(support={worst['support']}, "
              f"precision={worst['precision']:.4f}, recall={worst['recall']:.4f}, f1={worst['f1']:.4f})")

    # 可选：按图（pred_xxx.csv）快速查看每个图里最差类
    print("\n=== Per-Graph Quick Check (worst class by F1 in each graph) ===")
    for gname, d in sorted(per_graph.items()):
        rws, _ = per_class_metrics(d["y_true"], d["y_pred"])
        if not rws:
            continue
        rws = sorted(rws, key=lambda r: (r["f1"], -r["support"]))
        w = rws[0]
        print(f"{gname}: worst_class={w['class']} "
              f"(support={w['support']}, f1={w['f1']:.4f})")

if __name__ == "__main__":
    main()
