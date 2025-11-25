# -*- coding: utf-8 -*-
"""
Node-wise MLP for 4-class classification (No-CLI version)
本脚本移除了命令行参数，所有配置在“用户可修改配置”区域内直接定义。

与 Transformer 版本不同：本模型**只使用每个节点的特征 x**，不依赖 edge_index。

功能概要：
- 递归扫描 INPUT_DIR 下所有 *_pyg.pt 并加载为图（torch_geometric.data.Data）
- 按"图"做归纳式拆分：train / val / test（随机划分，受 SEED 控制）
- 仅用训练图拟合特征标准化（z-score），应用到全体图（可关闭）
- NodeMLP（多层全连接）进行 4 类节点分类（NUM_CLASSES 默认 4）
- 支持类别不平衡时的 class weight
- 早停（基于验证集 macro-F1），保存最优模型到 OUT_DIR
- 对每张图导出节点预测 CSV（node_id, y_true, y_pred, conf）

使用方式：
    1) 修改下方“用户可修改配置”中的 INPUT_DIR / OUT_DIR 等参数
    2) 运行：python TR_mlp_nocli.py
"""
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------
# 用户可修改配置（No-CLI）
# -------------------
INPUT_DIR         = "./graphs_out"          # 你的 *_pyg.pt 数据目录
OUT_DIR           = "./runs/mlp_from_pyg"  # 产物保存目录
TRAIN_RATIO       = 0.7
VAL_RATIO         = 0.15                   # TEST = 1 - TRAIN - VAL
SEED              = 42
BATCH_SIZE        = 1                      # 图级批量（每次一个或多个图；节点会自动拼接）
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
EPOCHS            = 200
PATIENCE          = 30
HIDDEN_DIM        = 128
MLP_LAYERS        = 2
DROPOUT           = 0.3
NUM_CLASSES       = 4                      # 目标分类数
USE_CLASS_WEIGHTS = True
STANDARDIZE       = True                   # 是否对特征做 z-score（基于训练集）

# -------------------
# Try import pyg data loader only (no conv layers needed)
# -------------------
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False
    raise RuntimeError("需要安装 PyTorch Geometric 的 Data/DataLoader 组件。"
                       "安装方式示例：pip install torch_geometric -f https://data.pyg.org/whl/torch-<TORCH_VER>+<CUDA>.html")

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
      1) {'x','y','edge_index'}  —— y 作为节点标签
      2) {'x','edge_index','node_type'} —— node_type 作为节点标签
    附加字段（node_ids/key 等）若存在会保留
    """
    obj = torch.load(p, map_location='cpu')
    if isinstance(obj, dict):
        d = {k: v for k, v in obj.items()}
    else:
        raise ValueError(f"{p} 内容不是字典类型")

    if 'y' in d:
        y = d['y']
    elif 'node_type' in d:
        y = d['node_type']
    else:
        raise KeyError(f"{p} 未找到标签字段 'y' 或 'node_type'")

    # 标准化字段名
    d['y'] = y
    if 'node_type' in d and 'y' not in d:
        d['y'] = d.pop('node_type')

    # 记录 key
    if 'key' not in d:
        d['key'] = p.stem.replace('_pyg', '')

    return d

def find_graph_files(input_dir: Path) -> List[Path]:
    return sorted(list(input_dir.rglob("*_pyg.pt")))

def to_pyg_data(d: Dict) -> "Data":
    x = d['x']
    y = d['y']
    edge_index = d.get('edge_index', None)
    node_ids = d.get('node_ids', None)

    # 确保为 tensor
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, y=y)
    # 兼容存储但不使用
    if edge_index is not None:
        data.edge_index = edge_index
    if node_ids is not None:
        if torch.is_tensor(node_ids):
            node_ids = node_ids.cpu().tolist()
        data.node_ids = node_ids

    data.key = d.get('key', 'graph')
    return data

def standardize_inplace(train_graphs: List["Data"], graphs: List["Data"]) -> Tuple[torch.Tensor, torch.Tensor]:
    """只用训练图的所有节点拟合 z-score，并原地应用到 graphs。返回 (mean, std)"""
    with torch.no_grad():
        train_feats = torch.cat([g.x for g in train_graphs], dim=0)
        mean = train_feats.mean(dim=0, keepdim=True)
        std  = train_feats.std(dim=0, unbiased=False, keepdim=True)
        std[std < 1e-6] = 1.0  # 避免除零
        for g in graphs:
            g.x.sub_(mean).div_(std)
    return mean.squeeze(0), std.squeeze(0)

def compute_class_weights(train_graphs: List["Data"], num_classes:int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for g in train_graphs:
        y = g.y.flatten()
        for c in range(num_classes):
            counts[c] += (y == c).sum()
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv * (counts.sum() / num_classes)
    return weights

def f1_macro(logits: torch.Tensor, targets: torch.Tensor, num_classes:int) -> float:
    preds = logits.argmax(dim=-1)
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        denom = (2*tp + fp + fn)
        f1 = 0.0 if denom == 0 else (2*tp) / denom
        f1s.append(f1)
    return float(np.mean(f1s))

# -------------------
# Model
# -------------------
class NodeMLP(nn.Module):
    """简单的节点级 MLP（不使用边）。depth 包含输出层。"""
    def __init__(self, in_dim: int, hidden: int, num_classes: int, depth: int = 2, dropout: float = 0.3):
        super().__init__()
        assert depth >= 1
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# -------------------
# Train / Eval
# -------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, device, num_classes:int) -> Tuple[float, float]:
    model.eval()
    all_logits = []
    all_targets = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x)            # 不使用 edge_index
        all_logits.append(logits.cpu())
        all_targets.append(batch.y.cpu())
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    acc = (logits.argmax(dim=-1) == targets).float().mean().item()
    f1  = f1_macro(logits, targets, num_classes)
    return acc, f1

def train_loop(model, train_loader, val_loader, device, optimizer, criterion, num_classes:int, epochs:int, patience:int, out_dir:Path):
    best_val_f1 = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x)        # 不使用 edge_index
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, val_f1 = evaluate(model, val_loader, device, num_classes)
        print(f"[Epoch {epoch:03d}] loss={total_loss:.4f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            # 保存最优
            torch.save({'state_dict': model.state_dict(),
                        'epoch': epoch}, out_dir / "model_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val F1={best_val_f1:.4f} (epoch {best_epoch}).")
                break

    # 加载最优
    ckpt = torch.load(out_dir / "model_best.pt", map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    return best_val_f1, best_epoch

# -------------------
# Main
# -------------------
def main():
    set_seed(SEED)

    input_dir = Path(INPUT_DIR)
    out_dir   = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_graph_files(input_dir)
    if len(files) == 0:
        raise FileNotFoundError(f"在 {input_dir} 下未找到 *_pyg.pt 文件")

    # 加载所有图
    dicts = [robust_load_graph_pt(p) for p in files]
    graphs = [to_pyg_data(d) for d in dicts]

    # 统计 in_dim 与（可选）数据的最大标签检查
    in_dim = graphs[0].x.size(-1)
    data_max_label = -1
    for g in graphs:
        if g.y.numel():
            data_max_label = max(data_max_label, int(g.y.max().item()))
    num_classes = int(NUM_CLASSES)
    if data_max_label >= 0 and (data_max_label + 1) > num_classes:
        print(f"[Warning] 数据中出现标签最大值 {data_max_label}，需要的类别数至少 {data_max_label+1}；当前设置为 NUM_CLASSES={num_classes}")

    # 拆分（按图）
    idx = list(range(len(graphs)))
    random.shuffle(idx)
    n_train = max(1, int(len(idx) * TRAIN_RATIO))
    n_val   = max(1, int(len(idx) * VAL_RATIO))
    n_test  = max(1, len(idx) - n_train - n_val)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    train_graphs = [graphs[i] for i in train_idx]
    val_graphs   = [graphs[i] for i in val_idx]
    test_graphs  = [graphs[i] for i in test_idx]

    # 标准化
    if STANDARDIZE:
        mean, std = standardize_inplace(train_graphs, graphs)
        torch.save({'mean': mean, 'std': std}, out_dir / "feature_norm.pt")
        print(f"Feature standardization: mean/std saved to {out_dir/'feature_norm.pt'}")

    # DataLoader（拼接节点）
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # 模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NodeMLP(in_dim, HIDDEN_DIM, num_classes, depth=MLP_LAYERS, dropout=DROPOUT).to(device)

    # 类别权重
    if USE_CLASS_WEIGHTS:
        weights = compute_class_weights(train_graphs, num_classes).to(device)
        print(f"class weights: {weights.cpu().numpy().round(4).tolist()}")
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 训练
    best_val_f1, best_epoch = train_loop(model, train_loader, val_loader, device, optimizer, criterion,
                                         num_classes, EPOCHS, PATIENCE, out_dir)

    # 测试
    test_acc, test_f1 = evaluate(model, test_loader, device, num_classes)
    print(f"[Test] acc={test_acc:.4f}  f1_macro={test_f1:.4f} (best_val_f1={best_val_f1:.4f} @epoch {best_epoch})")

    # 每图导出预测
    model.eval()
    with torch.no_grad():
        for g in graphs:
            gx = g.x.to(device)
            logits = model(gx)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            pred   = logits.argmax(dim=-1).cpu().numpy()
            y_true = g.y.cpu().numpy() if g.y is not None else np.full((g.x.size(0),), -1, dtype=np.int64)

            node_ids = getattr(g, 'node_ids', None)
            if node_ids is None:
                node_ids = list(range(g.x.size(0)))

            rows = []
            for i, nid in enumerate(node_ids):
                rows.append({
                    "node_id": int(nid),
                    "y_true": int(y_true[i]) if i < len(y_true) else -1,
                    "y_pred": int(pred[i]),
                    "conf": float(probs[i, pred[i]])
                })
            csv_path = out_dir / f"pred_{g.key}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["node_id","y_true","y_pred","conf"])
                w.writeheader()
                w.writerows(rows)

    print(f"完成。最优验证 F1={best_val_f1:.3f} (epoch {best_epoch})。所有产物保存在：{out_dir}")

if __name__ == "__main__":
    main()
