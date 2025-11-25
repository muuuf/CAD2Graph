# -*- coding: utf-8 -*-
"""
使用 Optuna 对图神经网络（基于 train_transformer_from_pyg_config.py）
进行超参数寻优 (HPO)。

此脚本将原始训练逻辑封装为 Optuna 的 'objective' 函数，
并自动搜索最佳的超参数组合。

**已按要求更新默认参数**:
--input_dir: ./graphs_out
--out_dir:   ./op_best
--n_trials:  50
"""

import argparse, sys, json, os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import random, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保安装了 optuna: pip install optuna
try:
    import optuna
except ImportError:
    print("Optuna not found. Please install it: pip install optuna", file=sys.stderr)
    sys.exit(1)

# 确保 PyG 可用
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import TransformerConv
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    print(f"Warning: PyTorch Geometric not loaded. Error: {e}", file=sys.stderr)


# -------------------------------------------------------------------
# 
#  从 train_transformer_from_pyg_config.py 复制的核心工具函数
#  (保持这部分与原脚本一致)
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

def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred == target).float().mean().item()

def f1_macro(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
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
    return float(np.mean(f1s))

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

@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> Tuple[float, float, torch.Tensor]:
    model.eval(); all_logits=[]; all_targets=[]
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        all_logits.append(logits); all_targets.append(batch.y)
    if not all_logits: return float('nan'), float('nan'), torch.empty(0)
    all_logits = torch.cat(all_logits, dim=0); all_targets = torch.cat(all_targets, dim=0)
    if all_targets.numel()==0: return float('nan'), float('nan'), all_logits.cpu()
    pred = all_logits.argmax(dim=-1); acc = accuracy(pred, all_targets)
    f1  = f1_macro(pred, all_targets, num_classes)
    return acc, f1, all_logits.cpu()

# -------------------------------------------------------------------
# 
#  Optuna Objective Function (封装了原始的 main 逻辑)
# 
# -------------------------------------------------------------------

def run_training(trial: optuna.trial.Trial, base_args: argparse.Namespace) -> float:
    """
    运行单个训练和评估 trial，被 Optuna 调用。
    
    Args:
        trial: Optuna trial 对象，用于建议超参数。
        base_args: 包含固定参数（如 input_dir, out_dir）的命名空间。
        
    Returns:
        float: 此 trial 的最佳验证集 F1-macro 分数。
    """
    
    # --- 1. 定义超参数搜索空间 ---
    # (你可以根据需要调整这里的范围)
    trial_params = {
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "heads": trial.suggest_categorical("heads", [4, 8, 16]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
        "use_class_weights": trial.suggest_categorical("use_class_weights", [True, False]),
        "standardize": trial.suggest_categorical("standardize", [True, False]),
    }
    
    # 将 base_args (固定参数) 和 trial_params (HPO参数) 合并
    args = argparse.Namespace(**vars(base_args))
    for k, v in trial_params.items():
        setattr(args, k, v)

    # --- 2. 设置此 trial 的唯一输出目录 ---
    # 所有产出 (模型, 日志) 将保存到这个独立目录
    out_dir = Path(args.out_dir) / f"trial_{trial.number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 3. 核心训练逻辑 (来自原始 main() 函数) ---
    
    if not PYG_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is required for training.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = Path(args.input_dir)
    
    pt_files = sorted(list(input_dir.rglob("*_pyg.pt")))
    if not pt_files: pt_files = sorted(list(input_dir.rglob("*.pt")))
    if not pt_files: raise RuntimeError(f"No .pt graphs found under {input_dir}")

    # 注意：为了 HPO 的一致性，数据加载和拆分应该在每个 trial 中重复
    # 并且依赖于固定的 `args.seed` 来确保拆分一致（如果 shuffle=True）
    # 在这里，我们先 shuffle 一次，然后按比例拆分
    all_graphs_dict = [robust_load_graph_pt(p) for p in pt_files]
    random.shuffle(all_graphs_dict) # 在拆分前 shuffle

    n_graphs = len(all_graphs_dict)
    n_train = int(n_graphs * args.train_ratio)
    n_val   = int(n_graphs * args.val_ratio)
    
    if n_train == 0 or n_val == 0:
        print(f"[Trial {trial.number}] Error: Not enough data for train/val split. Pruning.")
        raise optuna.TrialPruned()
        
    train_graphs = all_graphs_dict[:n_train]
    val_graphs   = all_graphs_dict[n_train:n_train+n_val]
    test_graphs  = all_graphs_dict[n_train+n_val:]
    print(f"[Trial {trial.number}] Split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    # 标准化 (在训练集上计算)
    mean = torch.zeros(1); std = torch.ones(1)
    if args.standardize:
        Xtr = torch.cat([g['x'] for g in train_graphs if g['x'].shape[0]>0], dim=0)
        if Xtr.shape[0]>0:
            _, mean, std = zscore(Xtr)
        else:
            print(f"[Trial {trial.number}] Warning: No nodes in training set for standardization.")
        
        for g in all_graphs_dict: # 应用到所有数据
            if g['x'].shape[0]>0:
                g['x'], _, _ = zscore(g['x'].float(), mean, std)

    # DataLoader
    to_data = lambda lst: [to_pyg_data(g) for g in lst]
    train_loader = DataLoader(to_data(train_graphs), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(to_data(val_graphs),   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(to_data(test_graphs),  batch_size=args.batch_size, shuffle=False)

    if not train_loader.dataset:
         print(f"[Trial {trial.number}] No training data after processing. Pruning.")
         raise optuna.TrialPruned()

    in_dim = int(train_loader.dataset[0].num_node_features)
    all_y  = torch.cat([d.y for d in train_loader.dataset + val_loader.dataset + test_loader.dataset if d.y.numel()>0])
    
    if all_y.numel() == 0:
        print(f"[Trial {trial.number}] No labels found in dataset. Pruning.")
        raise optuna.TrialPruned()
        
    num_classes = int(all_y.max().item()) + 1
    print(f"[Trial {trial.number}] Params: {trial_params}")
    print(f"[Trial {trial.number}] in_dim={in_dim}, num_classes={num_classes}")

    # 类别权重
    class_weights = None
    if args.use_class_weights:
        ytr = torch.cat([d.y for d in train_loader.dataset if d.y.numel()>0])
        if ytr.numel()>0:
            cnt = torch.bincount(ytr, minlength=num_classes).float()
            w = 1.0 / torch.clamp(cnt, min=1.0); w = w / w.mean()
            class_weights = w.to(device)
            print(f"[Trial {trial.number}] Using class weights: {w.cpu().numpy().round(2)}")

    # 模型, 损失函数, 优化器
    model = GraphTransformer2Layer(in_dim, args.hidden_dim, num_classes, heads=args.heads, dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练循环
    best_val_f1=-1.0; best_state=None; best_epoch=0; patience_left=args.patience; history=[]
    
    def eval_loader(loader):
        return evaluate(model, loader, device, num_classes)

    for epoch in range(1, args.epochs+1):
        model.train(); total_loss=0; total_nodes=0
        for batch in train_loader:
            if batch.num_nodes==0: continue
            batch = batch.to(device); opt.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y); loss.backward(); opt.step()
            total_loss += loss.item()*batch.num_nodes; total_nodes += batch.num_nodes
        
        if total_nodes==0: 
            print(f"[Trial {trial.number}] No nodes trained in epoch {epoch}. Skipping.")
            continue
            
        avg_loss = total_loss/total_nodes
        
        # 评估
        tr_acc,tr_f1,_ = eval_loader(train_loader)
        va_acc,va_f1,_ = eval_loader(val_loader)
        te_acc,te_f1,_ = eval_loader(test_loader)
        history.append({"epoch":epoch,"loss":avg_loss,"train_acc":tr_acc,"train_f1":tr_f1,"val_acc":va_acc,"val_f1":va_f1,"test_acc":te_acc,"test_f1":te_f1})
        
        # 检查是否为最佳
        if va_f1 > best_val_f1:
            best_val_f1=va_f1; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; patience_left=args.patience
        else:
            patience_left-=1
            if patience_left<=0:
                print(f"[Trial {trial.number}] Early stopping at epoch {epoch}.")
                break
        
        # --- 4. (关键) Optuna Pruning 和 Report ---
        # 向 Optuna 报告中间结果
        trial.report(va_f1, epoch)
        
        # 检查是否应该剪枝
        if trial.should_prune():
            print(f"[Trial {trial.number}] Pruned at epoch {epoch} due to poor performance.")
            raise optuna.TrialPruned()
        
        if epoch%20==0 or epoch==1: # 减少打印频率
            print(f"[T{trial.number:03d} E{epoch:03d}] loss={avg_loss:.4f} | va_f1={va_f1:.3f} (best={best_val_f1:.3f}) | te_f1={te_f1:.3f}")

    # --- 5. 保存此 Trial 的结果 ---
    if best_state is not None: 
        model.load_state_dict(best_state)
        torch.save({
            "state_dict": model.state_dict(),
            "in_dim": in_dim, "hidden_dim": args.hidden_dim, "num_classes": num_classes,
            "heads": args.heads, "dropout": args.dropout, "standardize": bool(args.standardize),
            "meta": {"best_val_f1": float(best_val_f1), "best_epoch": int(best_epoch)},
            "hparams": trial_params # 保存这组超参数
        }, out_dir / "best_model.pt")
    
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    
    print(f"[Trial {trial.number}] Done. Best val F1={best_val_f1:.3f} (epoch {best_epoch}). Out: {out_dir}")

    # --- 6. (关键) 返回要优化的指标 ---
    return best_val_f1

# -------------------------------------------------------------------
# 
#  HPO 主函数 (启动 Optuna Study)
# 
# -------------------------------------------------------------------

def main_hpo():
    parser = argparse.ArgumentParser(description="Optuna HPO for Graph Transformer")
    
    # --- 固定的路径和 HPO 设置 (已按要求设置默认值) ---
    parser.add_argument("--input_dir", default="./graphs_out", help="Path to input .pt graph data")
    parser.add_argument("--out_dir",   default="./op_best", help="Base output directory for ALL trials")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of HPO trials to run")
    
    # --- 其他参数 (保持不变) ---
    parser.add_argument("--storage", type=str, default="sqlite:///hpo_study.db", help="Optuna storage URL (e.g., sqlite:///study.db)")
    parser.add_argument("--study_name", type=str, default="graph_transformer_hpo", help="Optuna study name")
    
    # --- 固定的训练参数 (不参与 HPO) ---
    # 这些参数将传递给每个 trial，但 Optuna 不会改变它们
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=300, help="Max epochs per trial (early stopping enabled)")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience per trial")

    args = parser.parse_args()
    
    # --- 确保 PyG 可用 ---
    if not PYG_AVAILABLE:
        print("Error: PyTorch Geometric (PyG) is required but not found.", file=sys.stderr)
        print("Please install it (e.g., using official PyG instructions).", file=sys.stderr)
        sys.exit(1)

    # --- Optuna Study ---
    # 使用中位数剪枝器，在 10 个 epoch 后开始剪枝
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    
    study = optuna.create_study(
        direction="maximize",     # 目标：最大化 val_f1
        storage=args.storage,     # 存储 study 结果 (支持断点续传)
        study_name=args.study_name,
        pruner=pruner,
        load_if_exists=True       # 如果 study 已存在，则加载它
    )
    
    # 使用 lambda 函数将固定的 base_args 传递给 objective (run_training)
    objective_fn = lambda trial: run_training(trial, args)
    
    print(f"--- Starting HPO Study '{args.study_name}' ---")
    print(f"Storage: {args.storage}")
    print(f"Input Dir: {args.input_dir}")
    print(f"Output Dir: {args.out_dir}")
    print(f"Running for {args.n_trials} trials...")
    
    try:
        study.optimize(objective_fn, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("\n--- HPO Interrupted by user ---")

    # --- 打印 HPO 结果 ---
    print("\n--- HPO Finished ---")
    print(f"Study '{args.study_name}' results:")
    
    print(f"Best trial number: {study.best_trial.number}")
    print("  Best value (Val F1): {:.4f}".format(study.best_value))
    
    print("  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    - {k}: {v}")

    # 将最佳参数保存到 JSON 文件
    best_params_file = Path(args.out_dir) / "best_hpo_params.json"
    with best_params_file.open("w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
        
    print(f"\nBest params saved to: {best_params_file}")
    print(f"Find logs and model for best trial in: {Path(args.out_dir) / f'trial_{study.best_trial.number}'}")
    print("\nUse `optuna-dashboard` to visualize results (if using sqlite):")
    print(f"  $ optuna-dashboard {args.storage}")


if __name__ == "__main__":
    main_hpo()