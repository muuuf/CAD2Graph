# -*- coding: utf-8 -*-
"""
加载一个训练好的 NodeMLP 模型 (来自 TR_mlp_nocli.py)，
提取所有节点的中间层嵌入 (embeddings)，
并使用 t-SNE 进行二维可视化，按节点类别着色。

**已根据 TR_mlp_nocli.py 的内容进行更新**
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

# --- 导入 TR_mlp_nocli.py 中的配置 ---
# 我们需要这些配置来确保数据加载和模型构建一致
try:
    from TR_mlp_nocli import (
        INPUT_DIR, OUT_DIR, SEED, TRAIN_RATIO, VAL_RATIO,
        HIDDEN_DIM, MLP_LAYERS, DROPOUT, NUM_CLASSES, STANDARDIZE
    )
    print(f"已从 TR_mlp_nocli.py 加载配置：OUT_DIR={OUT_DIR}")
except ImportError:
    print("Error: 无法从 TR_mlp_nocli.py 导入配置。")
    print("请确保 TR_mlp_nocli.py 与此脚本位于同一目录中。")
    # 如果导入失败，使用硬编码的后备值
    INPUT_DIR         = "./graphs_out"
    OUT_DIR           = "./runs/mlp_from_pyg"
    SEED              = 42
    TRAIN_RATIO       = 0.7
    VAL_RATIO         = 0.15
    HIDDEN_DIM        = 128
    MLP_LAYERS        = 2
    DROPOUT           = 0.3
    NUM_CLASSES       = 4
    STANDARDIZE       = True
    print(f"Warning: 导入失败。使用后备配置：OUT_DIR={OUT_DIR}")


# --- 中文字体设置 ---
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False 
    print("已设置中文字体 'SimHei'。")
except Exception as e:
    print(f"Warning: 无法设置中文字体。标签可能显示不正确。Error: {e}")

# -------------------------------------------------------------------
# 
#  从 TR_mlp_nocli.py 复制的核心工具函数
# 
# -------------------------------------------------------------------

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def robust_load_graph_pt(p: Path) -> Dict:
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
    d['y'] = y
    if 'node_type' in d and 'y' not in d:
        d['y'] = d.pop('node_type')
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
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.long)
    data = Data(x=x, y=y)
    if edge_index is not None:
        data.edge_index = edge_index
    if node_ids is not None:
        if torch.is_tensor(node_ids):
            node_ids = node_ids.cpu().tolist()
        data.node_ids = node_ids
    data.key = d.get('key', 'graph')
    return data

def zscore(X: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """应用 z-score"""
    return (X - mean) / std

# -------------------------------------------------------------------
# 
#  从 TR_mlp_nocli.py 复制的 NodeMLP 模型定义
# 
# -------------------------------------------------------------------

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
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        辅助函数，用于提取第一个隐藏层的嵌入。
        """
        if not isinstance(self.net, nn.Sequential) or len(self.net) < 2:
            # 如果 depth=1, net 是一个 Linear 层, 没有隐藏嵌入
            return x 
        
        # 应用 Layer 0 (Linear)
        h = self.net[0](x)
        # 应用 Layer 1 (ReLU)
        h = self.net[1](h)
        return h

# -------------------------------------------------------------------
# 
#  修改后的嵌入提取函数 (用于 NodeMLP)
# 
# -------------------------------------------------------------------

@torch.no_grad()
def get_embeddings(model: NodeMLP, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在 loader 上的所有数据上运行 MLP 模型，
    并返回中间层嵌入 (layer1+ReLU 的输出) 和目标标签。
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    
    print("Generating node embeddings (from MLP hidden layer)...")
    for batch in loader:
        if batch.num_nodes == 0:
            continue
        batch = batch.to(device)
        
        # --- 获取中间层嵌入 ---
        # MLP 仅使用 batch.x
        h = model.get_embedding(batch.x)
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
#  t-SNE 可视化主函数 (已修改为加载 NodeMLP)
# 
# -------------------------------------------------------------------

def main_visualize():
    parser = argparse.ArgumentParser(description="Visualize node embeddings using t-SNE (MLP Version).")
    
    # --- 关键参数 (使用导入的配置作为默认值) ---
    parser.add_argument("--model_path", 
                        default=str(Path(OUT_DIR) / "model_best.pt"), 
                        help="MLP 模型的 'model_best.pt' 文件路径")
    parser.add_argument("--out_file", 
                        type=str, 
                        default=str(Path(OUT_DIR) / "tsne_mlp_all_split.png"),
                        help="保存 MLP t-SNE 绘图的输出文件路径")
    
    # --- 数据集参数 (使用导入的配置作为默认值) ---
    parser.add_argument("--input_dir", default=INPUT_DIR, 
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
        print(f"Error: MLP Model file not found at {model_path}", file=sys.stderr)
        print("请确保 --model_path 参数指向你训练好的 MLP (F1=0.344) 模型的 .pt 文件")
        sys.exit(1)
        
    print(f"Loading MLP model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    # --- 2. 加载和处理数据 (与训练时完全相同的逻辑) ---
    set_seed(args.seed) 
    
    input_dir_path = Path(args.input_dir)
    files = find_graph_files(input_dir_path)
    if len(files) == 0:
        raise FileNotFoundError(f"在 {input_dir_path} 下未找到 *_pyg.pt 文件")

    dicts = [robust_load_graph_pt(p) for p in files]
    graphs = [to_pyg_data(d) for d in dicts]
    
    in_dim = graphs[0].x.size(-1)

    # 拆分（按图），以确保我们能找到用于标准化的训练图
    idx = list(range(len(graphs)))
    random.shuffle(idx)
    n_train = max(1, int(len(idx) * args.train_ratio))
    n_val   = max(1, int(len(idx) * args.val_ratio))
    
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    train_graphs = [graphs[i] for i in train_idx]
    
    if args.dataset_split == 'train':
        graphs_to_plot = train_graphs
    elif args.dataset_split == 'val':
        graphs_to_plot = [graphs[i] for i in val_idx]
    elif args.dataset_split == 'test':
        graphs_to_plot = [graphs[i] for i in test_idx]
    else: # 'all'
        graphs_to_plot = graphs

    print(f"Loading {len(graphs_to_plot)} graphs from '{args.dataset_split}' split...")

    # --- 3. 标准化 (与训练时完全相同的逻辑) ---
    if STANDARDIZE:
        print("Applying standardization (calculated from training split)...")
        # 从训练脚本的 OUT_DIR 加载 mean/std
        norm_path = Path(OUT_DIR) / "feature_norm.pt"
        if not norm_path.exists():
            print(f"Error: 未找到标准化文件 {norm_path}。")
            print("请先运行 TR_mlp_nocli.py 训练脚本以生成此文件。")
            sys.exit(1)
        
        norm_data = torch.load(norm_path, map_location='cpu')
        mean = norm_data['mean'].unsqueeze(0) # 保持 [1, D] 维度
        std = norm_data['std'].unsqueeze(0)
        std[std < 1e-6] = 1.0
        
        for g in graphs_to_plot:
            g.x = zscore(g.x, mean, std) # 应用 z-score
    else:
        print("Skipping standardization (model was trained without it).")

    # --- 4. 实例化模型并加载状态 ---
    model = NodeMLP(
        in_dim=in_dim,
        hidden=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        depth=MLP_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    model.load_state_dict(checkpoint['state_dict'])
    print("MLP Model loaded successfully.")

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
    
    plt.title(f"t-SNE Visualization of Node Embeddings (MLP Baseline, {args.dataset_split} split)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    plt.savefig(args.out_file, dpi=300, bbox_inches='tight')
    
    print(f"Successfully saved MLP t-SNE plot to {args.out_file}")


if __name__ == "__main__":
    main_visualize()