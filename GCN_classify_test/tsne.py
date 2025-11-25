# -*- coding: utf-8 -*-
"""
加载一个训练好的 GraphTransformer 模型 (来自 Optuna HPO)，
提取所有节点的中间层嵌入 (embeddings)，
并使用 t-SNE 进行二维可视化，按节点类别着色。

**已按要求更新默认参数**:
--model_path:   ./op_best/trial_42/best_model.pt
--out_file:     ./tsne_all_split.png
--dataset_split: all
--sample_size:   10000

**修复**:
1. 将 TSNE 的参数从 'n_iter' 更改为 'max_iter' 以兼容新版 scikit-learn。
2. 移除 plt.legend() 中不兼容旧版 matplotlib 的 'markersfirst' 参数。
3. 修正 plt.cm.get_cmap() 的弃用警告。
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

# 确保 Scikit-learn 和 Matplotlib 可用
try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: scikit-learn and matplotlib are required.", file=sys.stderr)
    print("Please install them: pip install scikit-learn matplotlib", file=sys.stderr)
    sys.exit(1)

# --- 中文字体设置 (如果绘图时中文显示为方框，请取消注释并指定字体) ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一个常见的中文字体
    plt.rcParams['axes.unicode_minus'] = False 
    print("已设置中文字体 'SimHei'。")
except Exception as e:
    print(f"Warning: 无法设置中文字体。标签可能显示不正确。Error: {e}")
# -----------------------------------------------------------------


# -------------------------------------------------------------------
# 
#  从 train_transformer_from_pyg_config.py 复制的核心工具函数
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
#  新的嵌入提取函数
# 
# -------------------------------------------------------------------

@torch.no_grad()
def get_embeddings(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 loader 上的所有数据上运行模型，
    并返回中间层嵌入 (conv1 的输出) 和目标标签。
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("Generating node embeddings...")
    for batch in loader:
        if batch.num_nodes == 0:
            continue
        batch = batch.to(device)
        
        # --- 获取中间层嵌入 ---
        # (conv1): [N, in_dim] -> [N, hidden * heads]
        h = model.conv1(batch.x, batch.edge_index)
        h = F.elu(h)
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
#  t-SNE 可视化主函数
# 
# -------------------------------------------------------------------

def main_visualize():
    parser = argparse.ArgumentParser(description="Visualize node embeddings using t-SNE.")
    
    # --- 关键参数 (已按要求设置默认值) ---
    parser.add_argument("--model_path", default="./op_best/trial_42/best_model.pt", 
                        help="Path to the 'best_model.pt' file from an Optuna trial")
    parser.add_argument("--out_file", type=str, default="./tsne_all_split.png",
                        help="Path to save the output t-SNE plot PNG file")
    
    # --- 数据集参数 (必须与训练时一致!) ---
    parser.add_argument("--input_dir", default="./graphs_out", 
                        help="Path to input .pt graph data (must be same as used for training)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed used for data splitting (must be same as training)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Train ratio (must be same as training)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Validation ratio (must be same as training)")

    # --- t-SNE & 抽样参数 (已按要求设置默认值) ---
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. 加载模型 ---
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = GraphTransformer2Layer(
        in_dim=checkpoint['in_dim'],
        hidden=checkpoint['hidden_dim'],
        num_classes=checkpoint['num_classes'],
        heads=checkpoint['heads'],
        dropout=checkpoint['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['state_dict'])
    print("Model loaded successfully.")

    # --- 2. 加载和处理数据 (与训练时完全相同的逻辑) ---
    set_seed(args.seed) 
    
    input_dir = Path(args.input_dir)
    pt_files = sorted(list(input_dir.rglob("*_pyg.pt")))
    if not pt_files: pt_files = sorted(list(input_dir.rglob("*.pt")))
    if not pt_files: 
        print(f"Error: No .pt graphs found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    all_graphs_dict = [robust_load_graph_pt(p) for p in pt_files]
    random.shuffle(all_graphs_dict) # 使用了 seed，确保 shuffle 结果一致

    n_graphs = len(all_graphs_dict)
    n_train = int(n_graphs * args.train_ratio)
    n_val   = int(n_graphs * args.val_ratio)
    
    train_graphs = all_graphs_dict[:n_train]
    val_graphs   = all_graphs_dict[n_train:n_train+n_val]
    test_graphs  = all_graphs_dict[n_train+n_val:]
    
    if args.dataset_split == 'train':
        graphs_to_plot = train_graphs
    elif args.dataset_split == 'val':
        graphs_to_plot = val_graphs
    elif args.dataset_split == 'test':
        graphs_to_plot = test_graphs
    else: # 'all'
        graphs_to_plot = all_graphs_dict # <-- 此逻辑会选择整个数据集

    print(f"Loading {len(graphs_to_plot)} graphs from '{args.dataset_split}' split...")

    # --- 3. 标准化 (与训练时完全相同的逻辑) ---
    if checkpoint.get('standardize', False):
        print("Applying standardization (calculated from training split)...")
        # 始终使用训练集的均值和方差
        Xtr = torch.cat([g['x'] for g in train_graphs if g['x'].shape[0]>0], dim=0)
        if Xtr.shape[0]>0:
            _, mean, std = zscore(Xtr)
            for g in graphs_to_plot: # 对 'all' (即 all_graphs_dict) 中的每个图应用
                if g['x'].shape[0]>0:
                    g['x'], _, _ = zscore(g['x'].float(), mean, std)
    else:
        print("Skipping standardization (model was trained without it).")

    # --- 4. 创建 Loader 并获取嵌入 ---
    to_data = lambda lst: [to_pyg_data(g) for g in lst]
    loader = DataLoader(to_data(graphs_to_plot), batch_size=32, shuffle=False) # Batch size 不影响结果
    
    all_embeddings, all_labels = get_embeddings(model, loader, device)
    
    if all_embeddings.numel() == 0:
        print("Error: No embeddings were generated. Cannot plot.", file=sys.stderr)
        sys.exit(1)

    # --- 5. 抽样 ---
    N = all_embeddings.shape[0]
    if N > args.sample_size:
        print(f"Total nodes {N} > sample_size {args.sample_size}. Sampling...")
        # 设置随机种子以便抽样可复现
        np.random.seed(args.seed)
        indices = np.random.choice(N, args.sample_size, replace=False)
        sampled_embeddings = all_embeddings[indices].numpy()
        sampled_labels = all_labels[indices].numpy()
    else:
        print(f"Using all {N} nodes for t-SNE.")
        sampled_embeddings = all_embeddings.numpy()
        sampled_labels = all_labels.numpy()

    # --- 6. 运行 t-SNE ---
    print(f"Running t-SNE on {sampled_embeddings.shape[0]} nodes... (this may take a while)")
    tsne = TSNE(
        n_components=2, 
        perplexity=args.perplexity, 
        random_state=args.seed, 
        max_iter=1000,   # <-- 修复 1
        init='pca',      # 使用 PCA 初始化更稳定
        n_jobs=-1        # 使用所有核心
    )
    embeddings_2d = tsne.fit_transform(sampled_embeddings)
    print("t-SNE computation complete.")

    # --- 7. 绘图 ---
    print(f"Plotting and saving to {args.out_file}...")
    
    # 你的类别标签
    label_map = {
        0: "交通空间 (Class 0)",
        1: "公共空间 (Class 1)",
        2: "办公空间 (Class 2)",
        3: "辅助空间 (Class 3)"
    }
    
    plt.figure(figsize=(14, 10))
    
    unique_labels = np.unique(sampled_labels)
    
    # 使用一个好看的色盘
    try:
        # 修复 3: 使用 plt.get_cmap() 而不是 plt.cm.get_cmap()
        colors = plt.get_cmap('tab10', len(unique_labels)) 
    except Exception:
        colors = plt.get_cmap('viridis', len(unique_labels))

    
    for i, label_id in enumerate(unique_labels):
        # 找到所有属于这个类别的点的索引
        indices = (sampled_labels == label_id)
        # 绘制这些点的 2D 散点图
        plt.scatter(
            embeddings_2d[indices, 0], 
            embeddings_2d[indices, 1], 
            color=colors(i), 
            label=label_map.get(label_id, f"Class {label_id}"), 
            alpha=0.7, 
            s=10  # 点的大小
        )
    
    plt.title(f"t-SNE Visualization of Node Embeddings ({args.dataset_split} split)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    # 将图例放在图的外面，防止遮挡
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left") # <-- 修复 2: 移除 markersfirst 参数
    
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为图例留出空间
    
    # 保存图像
    plt.savefig(args.out_file, dpi=300, bbox_inches='tight')
    
    print(f"Successfully saved t-SNE plot to {args.out_file}")


if __name__ == "__main__":
    main_visualize()