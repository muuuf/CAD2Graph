# -*- coding: utf-8 -*-
"""
加载一个训练好的 GAT 模型 (来自 TR_GAT.py / TR.py)，
提取所有节点的中间层嵌入 (embeddings)，
并使用 t-SNE 进行二维可视化，按节点类别着色。

**已根据 TR_GAT.py (原 TR.py) 的内容进行更新**
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
    from torch_geometric.nn import GATConv # <--- 导入 GATConv
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    print(f"Warning: PyTorch Geometric not loaded. Error: {e}", file=sys.stderr)

# 确保 Scikit-learn 和 Matplotlib 可用
try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: scikit-learn and matplotlib are required.", file=sys.stderr)
    print("Please install them: pip install scikit-learn matplotlib", file=sys.stderr)
    sys.exit(1)

# --- 从 TR_GAT.py 复制的默认配置 ---
# (TR_GAT.py 使用 argparse defaults, 我们在这里设为后备值)
DEFAULT_INPUT_DIR = "./graphs_out"
DEFAULT_OUT_DIR   = "./runs/gat_from_pyg" # <--- GAT 默认输出
SEED              = 42
TRAIN_RATIO       = 0.7
VAL_RATIO         = 0.15
HIDDEN_DIM        = 64  # <--- GAT 默认
HEADS             = 16  # <--- GAT 默认
DROPOUT           = 0.2
STANDARDIZE       = True


# --- 中文字体设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    print("已设置中文字体 'SimHei'。")
except Exception as e:
    print(f"Warning: 无法设置中文字体。标签可能显示不正确。Error: {e}")

# -------------------------------------------------------------------
# 
#  从 TR_GAT.py (原 TR.py) 复制的核心工具函数
# 
# -------------------------------------------------------------------

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

def find_graph_files(input_dir: Path) -> List[Path]:
    """递归查找 _pyg.pt 或 .pt 文件"""
    pt_files = sorted(list(input_dir.rglob("*_pyg.pt")))
    if not pt_files:
        pt_files = sorted(list(input_dir.rglob("*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"在 {input_dir} 下未找到 *_pyg.pt 或 *.pt 文件")
    return pt_files

def to_pyg_data(g: Dict) -> "Data":
    """从 GAT 脚本复制的函数"""
    if not PYG_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is not installed.")
    data = Data(x=g['x'], edge_index=g['edge_index'], y=g['node_type'])
    data.node_ids = g['node_ids']
    data.key = g['meta']['key']
    return data

def zscore(X: torch.Tensor, mean: torch.Tensor=None, std: torch.Tensor=None):
    """从 GAT 脚本复制的函数 (可计算或应用)"""
    if mean is None or std is None:
        mean = X.mean(dim=0, keepdim=True)
        std = X.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std==0, torch.ones_like(std), std)
    # 返回 (tensor, mean, std)
    return (X-mean)/std, mean.squeeze(0), std.squeeze(0)

# -------------------------------------------------------------------
# 
#  从 TR_GAT.py (原 TR.py) 复制的 GAT 模型定义
# 
# -------------------------------------------------------------------

class GAT2Layer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, heads: int=4, dropout: float=0.5):
        super().__init__()
        if not PYG_AVAILABLE:
            raise RuntimeError("PyTorch Geometric is required for GATConv layers.")
        
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden * heads, num_classes, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def get_embedding(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        辅助函数，用于提取第一个 GAT 层的嵌入 (conv1 + elu)。
        """
        # 应用 Layer 1 (GATConv)
        h = self.conv1(x, edge_index)
        # 应用 Layer 1 (ELU)
        h = F.elu(h)
        return h

# -------------------------------------------------------------------
# 
#  修改后的嵌入提取函数 (用于 GAT2Layer)
# 
# -------------------------------------------------------------------

@torch.no_grad()
def get_embeddings(model: GAT2Layer, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 loader 上的所有数据上运行 GAT 模型，
    并返回中间层嵌入 (conv1+ELU 的输出) 和目标标签。
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("Generating node embeddings (from GAT hidden layer)...")
    for batch in loader:
        if batch.num_nodes == 0:
            continue
        batch = batch.to(device)
        
        # --- 获取中间层嵌入 ---
        # GAT 需要 x 和 edge_index
        h = model.get_embedding(batch.x, batch.edge_index) # <--- 修改点
        # ------------------------
        
        all_embeddings.append(h.cpu())
        all_labels.append(batch.y.cpu())
        
    if not all_embeddings:
        return torch.empty(0), torch.empty(0)
        
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    print(f"Generated {all_embeddings.shape[0]} embeddings of dim {all_embeddings.shape[1]}")
    return all_embeddings, all_labels

# -------------------------------------------------------------------
# 
#  t-SNE 可视化主函数 (已修改为加载 GAT)
# 
# -------------------------------------------------------------------

def main_visualize():
    parser = argparse.ArgumentParser(description="Visualize node embeddings using t-SNE (GAT Version).")
    
    # --- 关键参数 (使用 GAT 默认值) ---
    parser.add_argument("--model_path", 
                        default=str(Path(DEFAULT_OUT_DIR) / "best_model.pt"), # <--- 修改
                        help="GAT 模型的 'best_model.pt' 文件路径")
    parser.add_argument("--out_file", 
                        type=str, 
                        default=str(Path(DEFAULT_OUT_DIR) / "tsne_gat_all_split.png"), # <--- 修改
                        help="保存 GAT t-SNE 绘图的输出文件路径")
    
    # --- 数据集参数 (使用 GAT 默认值) ---
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, 
                        help="Path to input .pt graph data (必须与训练时一致)")
    parser.add_argument("--seed", type=int, default=SEED, 
                        help="Seed used for data splitting (必须与训练时一致)")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO,
                        help="Train ratio (必须与训练时一致)")
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO,
                        help="Validation ratio (必须与训练时一致)")

    # --- t-SNE & 抽样参数 ---
    parser.add_argument("--dataset_split", type=str, default="all",
                        choices=['train', 'val', 'test', 'all'],
                        help="Which data split to visualize")
    parser.add_argument("--sample_size", type=int, default=10000,
                        help="t-SNE is slow. Randomly sample this many nodes to plot.")
    parser.add_argument("--perplexity", type=int, default=30,
                        help="t-SNE perplexity parameter")

    args = parser.parse_args()
    
    if not PYG_AVAILABLE:
        print("Error: PyTorch Geometric (PyG) is required but not found.", file=sys.stderr)
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. 加载模型 ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: GAT Model file not found at {model_path}", file=sys.stderr)
        print("请确保 --model_path 参数指向你训练好的 GAT 模型的 'best_model.pt' 文件")
        sys.exit(1)
        
    print(f"Loading GAT model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # --- 2. 加载和处理数据 (与训练时完全相同的逻辑) ---
    set_seed(args.seed) 
    
    input_dir_path = Path(args.input_dir)
    files = find_graph_files(input_dir_path)
    
    # (使用 GAT 的函数)
    dicts = [robust_load_graph_pt(p) for p in files]
    graphs = [to_pyg_data(d) for d in dicts]

    # 拆分（按图），以确保我们能找到用于标准化的训练图
    idx = list(range(len(graphs)))
    random.shuffle(idx)
    n_train = max(1, int(len(idx) * args.train_ratio))
    n_val   = max(1, int(len(idx) * args.val_ratio))
    
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    if args.dataset_split == 'train':
        graphs_to_plot = [graphs[i] for i in train_idx]
    elif args.dataset_split == 'val':
        graphs_to_plot = [graphs[i] for i in val_idx]
    elif args.dataset_split == 'test':
        graphs_to_plot = [graphs[i] for i in test_idx]
    else: # 'all'
        graphs_to_plot = graphs

    print(f"Loading {len(graphs_to_plot)} graphs from '{args.dataset_split}' split...")

    # --- 3. 标准化 (从 GAT 检查点加载) ---
    STANDARDIZE = checkpoint.get('standardize', True)
    if STANDARDIZE:
        print("Applying standardization (loaded from model checkpoint)...")
        if 'standardize_mean' not in checkpoint or 'standardize_std' not in checkpoint:
            print(f"Error: 未在 {model_path} 中找到 'standardize_mean'/'standardize_std'。")
            print("请确保 GAT 模型是使用 TR.py 脚本训练并保存的。")
            sys.exit(1)
            
        mean = torch.tensor(checkpoint['standardize_mean'], dtype=torch.float32).unsqueeze(0)
        std = torch.tensor(checkpoint['standardize_std'], dtype=torch.float32).unsqueeze(0)
        
        for g in graphs_to_plot:
            if g.x.shape[0] > 0:
                # (使用 GAT 脚本的 zscore 函数)
                g.x, _, _ = zscore(g.x, mean, std) 
    else:
        print("Skipping standardization (model was trained without it).")

    # --- 4. 实例化模型并加载状态 ---
    # (从 GAT 检查点加载超参数)
    try:
        in_dim = checkpoint['in_dim']
        hidden_dim = checkpoint['hidden_dim']
        num_classes = checkpoint['num_classes']
        heads = checkpoint['heads']
    except KeyError as e:
        print(f"Error: 模型检查点 {model_path} 缺少关键超参数: {e}")
        sys.exit(1)

    model = GAT2Layer(
        in_dim=in_dim,
        hidden=hidden_dim,
        num_classes=num_classes,
        heads=heads,
        dropout=0.0 # eval 模式
    ).to(device)
    
    model.load_state_dict(checkpoint['state_dict'])
    print("GAT Model loaded successfully.")

    # --- 5. 创建 Loader 并获取嵌入 ---
    loader = DataLoader(graphs_to_plot, batch_size=32, shuffle=False)
    
    all_embeddings, all_labels = get_embeddings(model, loader, device)
    
    if all_embeddings.numel() == 0:
        print("Error: No embeddings were generated. Cannot plot.", file=sys.stderr)
        sys.exit(1)

    # --- 6. 抽样 ---
    N = all_embeddings.shape[0]
    if N > args.sample_size:
        print(f"Total nodes {N} > sample_size {args.sample_size}. Sampling...")
        np.random.seed(args.seed)
        indices = np.random.choice(N, args.sample_size, replace=False)
        sampled_embeddings = all_embeddings[indices].numpy()
        sampled_labels = all_labels[indices].numpy()
    else:
        print(f"Using all {N} nodes for t-SNE.")
        sampled_embeddings = all_embeddings.numpy()
        sampled_labels = all_labels.numpy()

    # --- 7. 运行 t-SNE ---
    print(f"Running t-SNE on {sampled_embeddings.shape[0]} nodes... (this may take a while)")
    tsne = TSNE(
        n_components=2, 
        perplexity=args.perplexity, 
        random_state=args.seed, 
        max_iter=1000,
        init='pca',
        n_jobs=-1
    )
    embeddings_2d = tsne.fit_transform(sampled_embeddings)
    print("t-SNE computation complete.")

    # --- 8. 绘图 ---
    print(f"Plotting and saving to {args.out_file}...")
    
    label_map = {
        0: "交通空间 (Class 0)",
        1: "公共空间 (Class 1)",
        2: "办公空间 (Class 2)",
        3: "辅助空间 (Class 3)"
    }
    
    plt.figure(figsize=(14, 10))
    unique_labels = np.unique(sampled_labels)
    
    try:
        colors = plt.get_cmap('tab10', len(unique_labels)) 
    except Exception:
        colors = plt.get_cmap('viridis', len(unique_labels))

    
    for i, label_id in enumerate(unique_labels):
        indices = (sampled_labels == label_id)
        plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1], 
            color=colors(i), 
            label=label_map.get(label_id, f"Class {label_id}"), 
            alpha=0.7, 
            s=10
        )
    
    plt.title(f"t-SNE Visualization of Node Embeddings (GAT Model, {args.dataset_split} split)", fontsize=16) # <--- 修改
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plt.savefig(args.out_file, dpi=300, bbox_inches='tight')
    
    print(f"Successfully saved GAT t-SNE plot to {args.out_file}")


if __name__ == "__main__":
    main_visualize()