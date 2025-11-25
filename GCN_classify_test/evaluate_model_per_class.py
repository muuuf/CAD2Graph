# -*- coding: utf-8 -*-
"""
加载一个训练好的 GraphTransformer 模型 (来自 Optuna HPO)，
并在测试集上运行评估，打印出详细的
分类报告 (per-class precision/recall/f1) 和混淆矩阵。
"""

import argparse, sys, json, os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保 PyG 可用
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import TransformerConv
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    print(f"Warning: PyTorch Geometric not loaded. Error: {e}", file=sys.stderr)

# 确保 Scikit-learn 可用
try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("scikit-learn not found. Please install it: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)


# -------------------------------------------------------------------
# 
#  从 train_transformer_from_pyg_config.py 复制的核心工具函数
#  (这部分必须与训练时使用的代码一致)
# 
# -------------------------------------------------------------------

def set_seed(seed:int):
    import numpy as np, random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def zscore(X: torch.Tensor, mean: torch.Tensor=None, std: torch.Tensor=None):
    if mean is None or std is None:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std==0, torch.ones_like(std), std)
    return (X-mean)/std, mean.squeeze(0), std.squeeze(0)

def robust_load_graph_pt(p: Path) -> Dict:
    g = torch.load(p, map_location='cpu')
    if 'x' not in g or 'edge_index' not in g:
        raise ValueError(f"Graph {p} missing x/edge_index")
    if 'node_type' in g: y = g['node_type']
    elif 'y' in g: y = g['y']
    else: raise ValueError(f"Graph {p} missing node_type/y")
    out = {
        'x': g['x'].float(),
        'edge_index': g['edge_index'].long(),
        'node_type': y.long(),
        'node_ids': g.get('node_ids', [str(i) for i in range(g['x'].shape[0])]),
        'meta': g.get('meta', {})
    }
    out['meta']['key'] = out['meta'].get('key', p.stem)
    return out

def to_pyg_data(g: Dict) -> "Data":
    if not PYG_AVAILABLE: raise RuntimeError("torch_geometric not found")
    data = Data(x=g['x'], edge_index=g['edge_index'], y=g['node_type'])
    data.node_ids = g['node_ids']; data.key = g['meta']['key']
    return data

class GraphTransformer2Layer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, heads: int=4, dropout: float=0.5):
        super().__init__()
        if not PYG_AVAILABLE: raise RuntimeError("PyG required")
        self.conv1 = TransformerConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = TransformerConv(hidden * heads, num_classes, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = torch.nn.functional.elu(x)
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); return x

# -------------------------------------------------------------------
# 
#  新的评估函数
# 
# -------------------------------------------------------------------

@torch.no_grad()
def get_test_predictions(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """在 loader 上的所有数据上运行模型并返回所有预测和目标"""
    model.eval()
    all_preds = []
    all_targets = []
    
    for batch in loader:
        if batch.num_nodes == 0:
            continue
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu())
        all_targets.append(batch.y.cpu())
        
    if not all_preds:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
        
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_preds, all_targets

def print_pretty_confusion_matrix(cm, labels):
    """打印一个格式化良好的混淆矩阵"""
    num_classes = len(labels)
    # 找到最长的标签或单元格所需的最大宽度
    cell_width = max(max(len(l) for l in labels), 6)
    
    # 打印标题行 (Predicted)
    header = f"{'Actual':<{cell_width}} | "
    for i, label in enumerate(labels):
        header += f"{label:>{cell_width}} "
    print("\n" + "—" * 20 + " Confusion Matrix " + "—" * 20)
    print(f" (Rows: Actual, Cols: Predicted)")
    print(header)
    print("-" * len(header))
    
    # 打印数据行 (Actual)
    for i in range(num_classes):
        row_str = f"{labels[i]:<{cell_width}} | "
        for j in range(num_classes):
            row_str += f"{cm[i, j]:>{cell_width}d} "
        print(row_str)
    print("-" * len(header))

# -------------------------------------------------------------------
# 
#  评估主函数
# 
# -------------------------------------------------------------------

def main_eval():
    parser = argparse.ArgumentParser(description="Evaluate a trained Graph Transformer model.")
    
    # --- 关键参数: 模型路径 ---
    parser.add_argument("--model_path", required=True, 
                        help="Path to the 'best_model.pt' file from an Optuna trial")
    
    # --- 数据集参数 (必须与训练时一致!) ---
    # 默认值已设为 HPO 脚本中的值
    parser.add_argument("--input_dir", default="./graphs_out", 
                        help="Path to input .pt graph data (must be same as used for training)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed used for data splitting (must be same as training)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Train ratio (must be same as training)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation ratio (must be same as training)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")

    args = parser.parse_args()
    
    if not PYG_AVAILABLE:
        print("Error: PyTorch Geometric (PyG) is required but not found.", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 加载模型 ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从 checkpoint 动态重建模型
    model = GraphTransformer2Layer(
        in_dim=checkpoint['in_dim'],
        hidden=checkpoint['hidden_dim'],
        num_classes=checkpoint['num_classes'],
        heads=checkpoint['heads'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("Model loaded successfully.")

    # --- 2. 加载和处理数据 (与训练时完全相同的逻辑) ---
    # 必须设置相同的种子以获得相同的 train/val/test 划分
    set_seed(args.seed) 
    
    input_dir = Path(args.input_dir)
    pt_files = sorted(list(input_dir.rglob("*_pyg.pt")))
    if not pt_files: pt_files = sorted(list(input_dir.rglob("*.pt")))
    if not pt_files: 
        print(f"Error: No .pt graphs found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    all_graphs_dict = [robust_load_graph_pt(p) for p in pt_files]
    random.shuffle(all_graphs_dict) # <--- 使用了 seed，确保 shuffle 结果一致

    n_graphs = len(all_graphs_dict)
    n_train = int(n_graphs * args.train_ratio)
    n_val   = int(n_graphs * args.val_ratio)
    
    train_graphs = all_graphs_dict[:n_train]
    val_graphs   = all_graphs_dict[n_train:n_train+n_val]
    test_graphs  = all_graphs_dict[n_train+n_val:]
    
    print(f"Data split (Seed={args.seed}): {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    # --- 3. 标准化 (与训练时完全相同的逻辑) ---
    mean = torch.zeros(1); std = torch.ones(1)
    # 检查模型是否在标准化数据上训练的
    if checkpoint.get('standardize', False):
        print("Applying standardization (calculated from training split)...")
        Xtr = torch.cat([g['x'] for g in train_graphs if g['x'].shape[0]>0], dim=0)
        if Xtr.shape[0]>0:
            _, mean, std = zscore(Xtr)
        
        # 仅对测试集应用变换
        for g in test_graphs:
            if g['x'].shape[0]>0:
                g['x'], _, _ = zscore(g['x'].float(), mean, std)
    else:
        print("Skipping standardization (model was trained without it).")

    # --- 4. 创建 Test Loader ---
    to_data = lambda lst: [to_pyg_data(g) for g in lst]
    test_loader = DataLoader(to_data(test_graphs), batch_size=args.batch_size, shuffle=False)
    
    if not test_loader.dataset:
        print("Error: No data in test loader.", file=sys.stderr)
        sys.exit(1)

    # --- 5. 获取预测结果 ---
    print("Running model on test set...")
    preds, targets = get_test_predictions(model, test_loader, device)
    
    if targets.numel() == 0:
        print("Error: No labels found in test set.", file=sys.stderr)
        sys.exit(1)

    # --- 6. 计算和打印详细指标 ---
    num_classes = checkpoint['num_classes']
    target_names = [f"Class {i}" for i in range(num_classes)]
    
    # 转换为 numpy 以便 scikit-learn 使用
    y_true = targets.numpy()
    y_pred = preds.numpy()
    
    # 打印 HPO 摘要
    print("\n" + "="*60)
    print(f"--- Evaluation Report for {model_path.parent.name} ---")
    print(f"Source Model: {model_path.name}")
    print(f"Trained for:  {checkpoint['meta']['best_epoch']} epochs")
    print(f"Best Val F1 (during HPO): {checkpoint['meta']['best_val_f1']:.4f}")
    print("="*60)
    
    # 打印分类报告
    print("\n" + "—" * 20 + " Classification Report " + "—" * 20)
    report = classification_report(
        y_true, 
        y_pred, 
        target_names=target_names, 
        digits=4,
        zero_division=0
    )
    print(report)
    
    # 打印混淆矩阵
    cm = confusion_matrix(
        y_true, 
        y_pred, 
        labels=list(range(num_classes)) # 确保所有类都显示
    )
    print_pretty_confusion_matrix(cm, target_names)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main_eval()