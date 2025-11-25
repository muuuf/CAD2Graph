# -*- coding: utf-8 -*-
"""
Optuna Hyperparameter Search for Graph Transformer
- Uses Optuna to find best hyperparameters for TransformerConv
- Tunes: LR, Weight Decay, Hidden Dims, Heads, Dropout
- Reports val_f1 and prunes unpromising trials.

Requirements: torch, numpy, torch_geometric, optuna

Usage:
    pip install optuna
    python train_transformer_optuna.py --input_dir ./Graph_out --out_dir ./runs/gcn_transformer_optuna --n_trials 50
"""

import argparse, json, os, random, math, csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import optuna
    from optuna.exceptions import TrialPruned
except Exception as e:
    import traceback, sys
    print("Failed to import Optuna:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import TransformerConv
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False
    print("Error: PyTorch Geometric (torch_geometric) is required.")

# -------------------
# CONFIG (defaults)
# -------------------
DEFAULT_INPUT_DIR = "./Graph_out"
DEFAULT_OUT_DIR   = "./runs/gcn_transformer_optuna" # New output dir
SEED              = 42
TRAIN_RATIO       = 0.7
VAL_RATIO         = 0.15
EPOCHS            = 200 # Max epochs *per trial*
EARLY_STOP_PATIENCE = 40
BATCH_SIZE        = 16
STANDARDIZE       = True
USE_CLASS_WEIGHTS = True
N_TRIALS          = 50 # Default number of Optuna trials

# -------------------
# Utilities (Unchanged)
# -------------------
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_graph_pt(p: Path) -> Dict:
    g = torch.load(p, map_location='cpu')
    if 'x' not in g or 'edge_index' not in g or 'node_type' not in g:
        raise ValueError(f"Graph file missing required keys: {p}")
    g.setdefault("meta", {})["key"] = g.get("meta", {}).get("key", p.stem)
    g.setdefault("node_ids", [str(i) for i in range(g["x"].shape[0])])
    return g

def to_pyg_data(g: Dict) -> "Data":
    x = g['x'].float()
    edge_index = g['edge_index'].long()
    y = g['node_type'].long()
    data = Data(x=x, edge_index=edge_index, y=y)
    data.node_ids = g['node_ids']
    data.key = g['meta']['key']
    return data

def zscore(X: torch.Tensor, mean: torch.Tensor=None, std: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
        recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))

# -------------------
# Model (Unchanged)
# -------------------
class GraphTransformer2Layer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, heads: int=4, dropout: float=0.5):
        super().__init__()
        self.conv1_dropout = dropout
        self.conv2_dropout = dropout
        self.model_dropout = dropout
        self.conv1 = TransformerConv(in_dim, hidden, heads=heads, dropout=self.conv1_dropout, concat=True)
        self.conv2 = TransformerConv(hidden * heads, num_classes, heads=1, dropout=self.conv2_dropout, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.model_dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# -------------------
# Evaluation (Unchanged)
# -------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes) -> Tuple[float, float, torch.Tensor]:
    model.eval()
    all_logits = []
    all_targets = []
    for batch in loader:
        if batch.num_nodes == 0: continue
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        all_logits.append(logits)
        all_targets.append(batch.y)
    
    if not all_logits:
        return float('nan'), float('nan'), torch.empty(0)

    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    if all_targets.numel() == 0:
        return float('nan'), float('nan'), all_logits.cpu()
        
    pred = all_logits.argmax(dim=-1)
    acc = accuracy(pred, all_targets)
    f1 = f1_macro(pred, all_targets, num_classes)
    
    return acc, f1, all_logits.cpu()

# -------------------
# Optuna Objective
# -------------------

def objective(
    trial: optuna.trial.Trial,
    device: torch.device,
    in_dim: int,
    num_classes: int,
    class_weights: torch.Tensor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args: argparse.Namespace
) -> float:
    
    # 1. Define Hyperparameters to tune
    hp = {
        'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        'hidden_dim': trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        'heads': trial.suggest_categorical("heads", [2, 4, 8]),
        'dropout': trial.suggest_float("dropout", 0.1, 0.6)
    }

    # 2. Initialize Model, Loss, Optimizer
    model = GraphTransformer2Layer(
        in_dim=in_dim,
        hidden=hp['hidden_dim'],
        num_classes=num_classes,
        heads=hp['heads'],
        dropout=hp['dropout']
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=hp['lr'], 
        weight_decay=hp['weight_decay']
    )

    # 3. Training loop
    best_val_f1 = -1.0
    patience_left = args.patience
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_nodes = 0
        for batch in train_loader:
            if batch.num_nodes == 0: continue
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_nodes
            total_nodes += batch.num_nodes
        
        if total_nodes == 0: continue
        avg_loss = total_loss / total_nodes

        # Evaluation
        _, tr_f1, _ = evaluate(model, train_loader, device, num_classes)
        va_acc, va_f1, _ = evaluate(model, val_loader, device, num_classes)
        _, te_f1, _ = evaluate(model, test_loader, device, num_classes)
        
        # 4. Optuna Pruning
        trial.report(va_f1, epoch)
        if trial.should_prune():
            raise TrialPruned()

        # Early stopping
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # 5. Return the metric to optimize
    return best_val_f1

# -------------------
# Main Function
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out_dir",   default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val_ratio",   type=float, default=VAL_RATIO)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_trials", type=int, default=N_TRIALS) # New
    parser.add_argument("--use_class_weights", action="store_true", default=USE_CLASS_WEIGHTS)
    parser.add_argument("--no_class_weights", action="store_false", dest="use_class_weights")
    parser.add_argument("--standardize", action="store_true", default=STANDARDIZE)
    parser.add_argument("--no_standardize", action="store_false", dest="standardize")
    args = parser.parse_args()

    if not PYG_AVAILABLE:
        raise RuntimeError("This script requires PyTorch Geometric.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === 1. Load Data (Once) ===
    graphs_dir = Path(args.input_dir) / "graphs"
    pt_files = sorted(graphs_dir.glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"No .pt graphs found under {graphs_dir}")
    
    all_graphs_dict = [load_graph_pt(p) for p in pt_files]
    random.shuffle(all_graphs_dict)

    n_graphs = len(all_graphs_dict)
    n_train = int(n_graphs * args.train_ratio)
    n_val = int(n_graphs * args.val_ratio)
    
    train_graphs_dict = all_graphs_dict[:n_train]
    val_graphs_dict   = all_graphs_dict[n_train : n_train + n_val]
    test_graphs_dict  = all_graphs_dict[n_train + n_val:]

    print(f"Dataset split: {len(train_graphs_dict)} train, {len(val_graphs_dict)} val, {len(test_graphs_dict)} test graphs.")

    mean, std = torch.zeros(1), torch.ones(1)
    if args.standardize:
        X_train_all = torch.cat([g['x'].float() for g in train_graphs_dict if g['x'].shape[0] > 0], dim=0)
        if X_train_all.shape[0] > 0:
            _, mean, std = zscore(X_train_all)
        for g in all_graphs_dict:
            if g['x'].shape[0] > 0:
                g['x'], _, _ = zscore(g['x'].float(), mean, std)
    
    train_datas = [to_pyg_data(g) for g in train_graphs_dict]
    val_datas   = [to_pyg_data(g) for g in val_graphs_dict]
    test_datas  = [to_pyg_data(g) for g in test_graphs_dict]
    
    in_dim = int(train_datas[0].num_node_features)
    all_y = torch.cat([d.y for d in train_datas + val_datas + test_datas if d.y.numel() > 0])
    num_classes = int(all_y.max().item()) + 1

    train_loader = DataLoader(train_datas, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_datas, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_datas, batch_size=args.batch_size, shuffle=False)

    class_weights = None
    if args.use_class_weights:
        y_train_all = torch.cat([d.y for d in train_datas if d.y.numel() > 0])
        if y_train_all.numel() > 0:
            class_counts = torch.bincount(y_train_all, minlength=num_classes).float()
            weights = 1.0 / torch.clamp(class_counts, min=1.0)
            weights = weights / weights.mean()
            class_weights = weights.to(device)

    # === 2. Run Optuna Study ===
    
    # We use a lambda function to pass all the pre-loaded data to the objective
    objective_with_data = lambda trial: objective(
        trial, device, in_dim, num_classes, class_weights,
        train_loader, val_loader, test_loader, args
    )

    # Create a study. 'direction="maximize"' means we want to maximize the val_f1
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    
    try:
        study.optimize(objective_with_data, n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("Optuna study interrupted by user.")

    # === 3. Print Results ===
    print("\n" + "="*30)
    print("Optuna Study Finished.")
    
    if study.best_trial:
        print(f"Best trial: {study.best_trial.number}")
        print(f"  Best val_f1: {study.best_trial.value:.4f}")
        print("  Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best params to a file
        best_params_file = out_dir / "best_params.json"
        with open(best_params_file, "w") as f:
            json.dump(study.best_trial.params, f, indent=2)
        print(f"Best parameters saved to {best_params_file}")

    else:
        print("No trials were completed.")
    print("="*30)

if __name__ == "__main__":
    main()