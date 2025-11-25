# -*- coding: utf-8 -*-
"""
Graph Transformer (TransformerConv) for Node Classification - GRAPH-LEVEL (Inductive) SPLIT
兼容你批量构图脚本的输出（<base>_pyg.pt），自动递归扫描 input_dir 下所有 *_pyg.pt。

- 支持两种 .pt 格式：
  1) 我们的批处理导出：包含 x, y, edge_index  （本脚本会把 y 视为 node_type）
  2) 原版格式：包含 x, edge_index, node_type  （保持兼容）

- 将“整图”划分为 train/val/test
- 仅用训练集拟合特征标准化并应用到全体
- 早停并保存最优模型
- 每个图导出 per-node 预测 CSV

Usage:
    python train_transformer_from_pyg.py --input_dir C:/data/graphs_out --out_dir ./runs/gt_from_pyg
    
MODIFIED:
- Added Confusion Matrix and Per-Class F1-Scores to the final evaluation.
"""

import argparse, json, os, random, math, csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to use PyTorch Geometric if available
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import TransformerConv
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False
    print("Error: PyTorch Geometric (torch_geometric) is required but not found.")
    print("Please install it, e.g., via: pip install torch_geometric -f https://data.pyg.org/whl/torch-<YOUR_TORCH_VER>+<CUDA>.html")

# -------------------
# CONFIG (defaults)
# -------------------
DEFAULT_INPUT_DIR = "./graphs_out"       # 根目录：包含很多 <base>/<base>_pyg.pt
DEFAULT_OUT_DIR   = "./runs/gt_from_pyg"
SEED              = 42
TRAIN_RATIO       = 0.7
VAL_RATIO         = 0.15
HIDDEN_DIM        = 64
HEADS             = 16
DROPOUT           = 0.2
LR                = 5e-3
WEIGHT_DECAY      = 1e-4
EPOCHS            = 300
EARLY_STOP_PATIENCE = 40
USE_CLASS_WEIGHTS = True
BATCH_SIZE        = 16
STANDARDIZE       = True  # z-score based on TRAIN graphs

# -------------------
# Utilities
# -------------------
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def robust_load_graph_pt(p: Path) -> Dict:
    """
    兼容两种 .pt：
    - 我们的 <base>_pyg.pt: keys = {x, y, edge_index, (可无 node_ids)}
    - 原版: keys = {x, edge_index, node_type, (node_ids)}
    统一返回 dict：{x, edge_index, node_type, node_ids, meta}
    """
    g = torch.load(p, map_location='cpu')

    # 标准化字段
    if 'x' not in g or 'edge_index' not in g:
        raise ValueError(f"Graph {p} missing x/edge_index")
    # 兼容 y vs node_type
    if 'node_type' in g:
        y = g['node_type']
    elif 'y' in g:
        y = g['y']
    else:
        raise ValueError(f"Graph {p} missing node_type/y")

    out = {
        'x': g['x'].float(),
        'edge_index': g['edge_index'].long(),
        'node_type': y.long(),
        'node_ids': g.get('node_ids', [str(i) for i in range(g['x'].shape[0])]),
        'meta': g.get('meta', {})
    }
    # 给 key（图名）
    out['meta']['key'] = out['meta'].get('key', p.stem)
    return out

def to_pyg_data(g: Dict) -> "Data":
    if not PYG_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is not installed.")
    data = Data(x=g['x'], edge_index=g['edge_index'], y=g['node_type'])
    data.node_ids = g['node_ids']
    data.key = g['meta']['key']
    return data

def zscore(X: torch.Tensor, mean: torch.Tensor=None, std: torch.Tensor=None):
    if mean is None or std is None:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std==0, torch.ones_like(std), std)
    return (X-mean)/std, mean.squeeze(0), std.squeeze(0)

def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred == target).float().mean().item()

def calculate_f1_scores(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Tuple[float, List[float]]:
    """
    [MODIFIED]
    计算 Macro-F1 和 Per-Class F1
    返回: (macro_f1, per_class_f1_list)
    """
    f1s = []
    for c in range(num_classes):
        tp = ((pred==c) & (target==c)).sum().item()
        fp = ((pred==c) & (target!=c)).sum().item()
        fn = ((pred!=c) & (target==c)).sum().item()
        if tp+fp==0 or tp+fn==0:
            f1 = 0.0
        else:
            precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
            recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)), f1s

@torch.no_grad()
def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    [NEW]
    计算混淆矩阵. 假设 pred/target 都在 CPU 上.
    返回: (num_classes, num_classes) 张量 (Rows: True, Cols: Pred)
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(target.view(-1), pred.view(-1)):
        cm[t.long(), p.long()] += 1
    return cm.cpu()

# -------------------
# Model
# -------------------
class GraphTransformer2Layer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, heads: int=4, dropout: float=0.5):
        super().__init__()
        if not PYG_AVAILABLE:
            raise RuntimeError("PyTorch Geometric is required for TransformerConv layers.")
        self.conv1 = TransformerConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = TransformerConv(hidden * heads, num_classes, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# -------------------
# Eval
# -------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> Tuple[float, float, List[float], torch.Tensor, torch.Tensor]:
    """
    [MODIFIED]
    返回: (acc, macro_f1, per_class_f1, confusion_matrix, logits)
    """
    model.eval()
    all_logits = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        if batch.num_nodes == 0:
            continue
        logits = model(batch.x, batch.edge_index)
        all_logits.append(logits)
        all_targets.append(batch.y)
        
    if not all_logits:
        empty_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        return float('nan'), float('nan'), [float('nan')] * num_classes, empty_cm, torch.empty(0)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    if all_targets.numel() == 0:
        empty_cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
        return float('nan'), float('nan'), [float('nan')] * num_classes, empty_cm, all_logits.cpu()
        
    # 移动到 CPU 进行指标计算
    all_logits_cpu = all_logits.cpu()
    all_targets_cpu = all_targets.cpu()
    pred_cpu = all_logits_cpu.argmax(dim=-1)

    acc = accuracy(pred_cpu, all_targets_cpu)
    macro_f1, per_class_f1 = calculate_f1_scores(pred_cpu, all_targets_cpu, num_classes) # MODIFIED
    cm = confusion_matrix(pred_cpu, all_targets_cpu, num_classes) # NEW
    
    return acc, macro_f1, per_class_f1, cm, all_logits_cpu # MODIFIED

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR,
                        help="根目录（包含很多 <base>/<base>_pyg.pt），将递归扫描 *_pyg.pt 与 *.pt")
    parser.add_argument("--out_dir",   default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--use_class_weights", action="store_true", default=USE_CLASS_WEIGHTS)
    parser.add_argument("--no_class_weights", action="store_false", dest="use_class_weights")
    parser.add_argument("--standardize", action="store_true", default=STANDARDIZE)
    parser.add_argument("--no_standardize", action="store_false", dest="standardize")
    args = parser.parse_args()

    if not PYG_AVAILABLE:
        raise RuntimeError("This script requires PyTorch Geometric. Please install it.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 递归扫描 .pt
    # 优先 *_pyg.pt；若没有，再用 *.pt（兼容旧格式）
    pt_files = sorted(list(input_dir.rglob("*_pyg.pt")))
    if not pt_files:
        pt_files = sorted(list(input_dir.rglob("*.pt")))
    if not pt_files:
        raise RuntimeError(f"No .pt graphs found under {input_dir}")

    # 2) 读取全部图
    all_graphs_dict = [robust_load_graph_pt(p) for p in pt_files]
    random.shuffle(all_graphs_dict)

    # 3) 划分整图
    n_graphs = len(all_graphs_dict)
    n_train = int(n_graphs * args.train_ratio)
    n_val = int(n_graphs * args.val_ratio)
    train_graphs_dict = all_graphs_dict[:n_train]
    val_graphs_dict   = all_graphs_dict[n_train : n_train + n_val]
    test_graphs_dict  = all_graphs_dict[n_train + n_val:]
    print(f"Dataset split: {len(train_graphs_dict)} train, {len(val_graphs_dict)} val, {len(test_graphs_dict)} test")

    # 4) 标准化（只用训练图拟合）
    mean, std = torch.zeros(1), torch.ones(1)
    if args.standardize:
        X_train_all = torch.cat([g['x'].float() for g in train_graphs_dict if g['x'].shape[0] > 0], dim=0)
        if X_train_all.shape[0] > 0:
            _, mean, std = zscore(X_train_all)
        for g in all_graphs_dict:
            if g['x'].shape[0] > 0:
                g['x'], _, _ = zscore(g['x'].float(), mean, std)

    # 5) 转 PyG Data
    train_datas = [to_pyg_data(g) for g in train_graphs_dict]
    val_datas   = [to_pyg_data(g) for g in val_graphs_dict]
    test_datas  = [to_pyg_data(g) for g in test_graphs_dict]

    if not train_datas:
        raise RuntimeError("No training graphs found or loaded.")
        
    in_dim = int(train_datas[0].num_node_features)
    all_y_list = [d.y for d in train_datas + val_datas + test_datas if d.y is not None and d.y.numel() > 0]
    if not all_y_list:
        raise RuntimeError("No labels (y) found in any graph.")
    all_y = torch.cat(all_y_list)
    num_classes = int(all_y.max().item()) + 1
    print(f"in_dim={in_dim}, num_classes={num_classes}")

    # 6) DataLoader
    train_loader = DataLoader(train_datas, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_datas, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_datas, batch_size=args.batch_size, shuffle=False)

    # 7) 类别权重（仅训练集）
    class_weights = None
    if args.use_class_weights:
        y_train_all_list = [d.y for d in train_datas if d.y is not None and d.y.numel() > 0]
        if y_train_all_list:
            y_train_all = torch.cat(y_train_all_list)
            if y_train_all.numel() > 0:
                class_counts = torch.bincount(y_train_all, minlength=num_classes).float()
                weights = 1.0 / torch.clamp(class_counts, min=1.0)
                weights = weights / weights.mean()
                class_weights = weights.to(device)

    # 8) 模型
    model = GraphTransformer2Layer(in_dim, args.hidden_dim, num_classes,
                                   heads=args.heads, dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 9) 训练 + 早停
    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0
    history = []
    patience_left = args.patience

    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0
        total_nodes = 0
        for batch in train_loader:
            if batch.num_nodes == 0: 
                continue
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes
        if total_nodes == 0:
            print(f"[Epoch {epoch:03d}] No nodes in training batches, skipping...")
            continue
        avg_loss = total_loss / total_nodes

        # [MODIFIED] 更新拆包以匹配 evaluate 的新返回
        tr_acc, tr_f1, _, _, _ = evaluate(model, train_loader, device, num_classes)
        va_acc, va_f1, _, _, _ = evaluate(model, val_loader, device, num_classes)
        te_acc, te_f1, _, _, _ = evaluate(model, test_loader, device, num_classes)

        history.append({"epoch": epoch, "loss": avg_loss,
                        "train_acc": tr_acc, "train_f1": tr_f1,
                        "val_acc": va_acc, "val_f1": va_f1,
                        "test_acc": te_acc, "test_f1": te_f1})

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f} | "
                  f"train_f1={tr_f1:.3f} val_f1={va_f1:.3f} test_f1={te_f1:.3f}")

    # 10) [MODIFIED] 恢复最优, 最终评估, 并保存
    if best_state is not None:
        model.load_state_dict(best_state)
        
    print(f"\n--- Final Evaluation on Test Set (using best model from epoch {best_epoch}) ---")
    test_acc, test_macro_f1, test_f1_per_class, test_cm, _ = evaluate(
        model, test_loader, device, num_classes
    )
    
    print(f"Best Epoch (by val_f1): {best_epoch}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_macro_f1:.4f}")
    
    print("\nTest Per-Class F1-Scores:")
    if num_classes == 0:
        print("  (No classes found)")
    for i, f1 in enumerate(test_f1_per_class):
        print(f"  Class {i}: {f1:.4f}")
        
    print("\nTest Confusion Matrix (Rows: True, Cols: Pred):")
    # 使用 numpy 格式化打印张量
    cm_str = np.array2string(test_cm.cpu().numpy(), 
                             prefix="       ", 
                             separator=', ')
    print(f"       {cm_str}\n")

    torch.save({
        "state_dict": model.state_dict(), "in_dim": in_dim, "hidden_dim": args.hidden_dim,
        "num_classes": num_classes, "heads": args.heads,
        "standardize": bool(args.standardize),
        "standardize_mean": mean.cpu().numpy().tolist(),
        "standardize_std": std.cpu().numpy().tolist(),
        # [NEW] 保存详细的测试指标
        "meta": {"best_val_f1": float(best_val_f1), "best_epoch": int(best_epoch),
                 "test_acc": float(test_acc), 
                 "test_macro_f1": float(test_macro_f1),
                 "test_f1_per_class": [float(f) for f in test_f1_per_class],
                 "test_confusion_matrix": test_cm.cpu().numpy().tolist()
                }
    }, out_dir / "best_model.pt")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 11) 每图预测
    print("Generating per-graph predictions...")
    for g_dict in all_graphs_dict:
        if g_dict['x'].shape[0] == 0: 
            continue
        data = to_pyg_data(g_dict).to(device)
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            pred  = probs.argmax(axis=1)
        rows = []
        node_ids = g_dict['node_ids']
        node_ys = g_dict['node_type']
        for i, nid in enumerate(node_ids):
            rows.append({
                "node_id": nid,
                "y_true": int(node_ys[i].item()),
                "y_pred": int(pred[i]),
                "conf": float(probs[i, pred[i]])
            })
        out_csv = out_dir / f"pred_{g_dict['meta']['key']}.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["node_id","y_true","y_pred","conf"])
            w.writeheader()
            w.writerows(rows)
    
    # [MODIFIED] 更新最终的打印信息
    print(f"Done. Best val F1={best_val_f1:.3f} (epoch {best_epoch}). "
          f"Final Test Macro-F1={test_macro_f1:.4f}. Artifacts saved in: {out_dir}")

if __name__ == "__main__":
    main()