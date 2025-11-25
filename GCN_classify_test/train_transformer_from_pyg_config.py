
# -*- coding: utf-8 -*-
"""
Training script that loads hyperparameters from an external config (hp_config.py)
and overrides/merges with CLI args. Falls back to defaults if config not provided.

It wraps the previously provided train_transformer_from_pyg.py core logic.
"""

import argparse, importlib.util, sys, json, os
from pathlib import Path

# Import the original training script content (inlined minimal wrapper):
from typing import List, Tuple, Dict, Optional
import random, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import TransformerConv
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False

# ------------------- core utils copied -------------------
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

# ------------------- config loader -------------------
def load_hp_config(path: str, preset: str) -> dict:
    spec = importlib.util.spec_from_file_location("hp_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "get_config"):
        raise AttributeError("Config file must define get_config(preset:str)->dict")
    cfg = mod.get_config(preset)
    if not isinstance(cfg, dict):
        raise TypeError("get_config must return dict")
    return cfg

def apply_config_to_args(args, cfg: dict):
    # Only override args that are not explicitly set via CLI (we can't detect directly).
    # Strategy: config sets defaults, but CLI still can override by passing flags.
    for k, v in cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)
    return args

# ------------------- main -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir",   required=True)

    # core hparams (will be overridden by config if provided)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--no_class_weights", action="store_false", dest="use_class_weights")
    parser.add_argument("--no_standardize", action="store_false", dest="standardize")

    # config file & preset
    parser.add_argument("--config-file", help="Path to hp_config.py", default=None)
    parser.add_argument("--config-preset", help="Preset name in hp_config.py", default="baseline")
    args = parser.parse_args()

    # load config if provided
    if args.config_file:
        cfg = load_hp_config(args.config_file, args.config_preset)
        args = apply_config_to_args(args, cfg)

    if not PYG_AVAILABLE:
        raise RuntimeError("PyTorch Geometric is required.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = Path(args.input_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(list(input_dir.rglob("*_pyg.pt")))
    if not pt_files: pt_files = sorted(list(input_dir.rglob("*.pt")))
    if not pt_files: raise RuntimeError(f"No .pt graphs found under {input_dir}")

    all_graphs_dict = [robust_load_graph_pt(p) for p in pt_files]
    random.shuffle(all_graphs_dict)

    n_graphs = len(all_graphs_dict)
    n_train = int(n_graphs * args.train_ratio)
    n_val   = int(n_graphs * args.val_ratio)
    train_graphs = all_graphs_dict[:n_train]
    val_graphs   = all_graphs_dict[n_train:n_train+n_val]
    test_graphs  = all_graphs_dict[n_train+n_val:]
    print(f"Split: {len(train_graphs)} train, {len(val_graphs)} val, {len(test_graphs)} test")

    # standardize on train
    mean = torch.zeros(1); std = torch.ones(1)
    if args.standardize:
        Xtr = torch.cat([g['x'] for g in train_graphs if g['x'].shape[0]>0], dim=0)
        if Xtr.shape[0]>0:
            _, mean, std = zscore(Xtr)
        for g in all_graphs_dict:
            if g['x'].shape[0]>0:
                g['x'], _, _ = zscore(g['x'].float(), mean, std)

    # DataLoader
    to_data = lambda lst: [to_pyg_data(g) for g in lst]
    train_loader = DataLoader(to_data(train_graphs), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(to_data(val_graphs),   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(to_data(test_graphs),  batch_size=args.batch_size, shuffle=False)

    in_dim = int(train_loader.dataset[0].num_node_features)
    all_y  = torch.cat([d.y for d in train_loader.dataset + val_loader.dataset + test_loader.dataset if d.y.numel()>0])
    num_classes = int(all_y.max().item()) + 1
    print(f"in_dim={in_dim}, num_classes={num_classes}")

    # class weights
    class_weights = None
    if args.use_class_weights:
        ytr = torch.cat([d.y for d in train_loader.dataset if d.y.numel()>0])
        if ytr.numel()>0:
            cnt = torch.bincount(ytr, minlength=num_classes).float()
            w = 1.0 / torch.clamp(cnt, min=1.0); w = w / w.mean()
            class_weights = w.to(device)

    model = GraphTransformer2Layer(in_dim, args.hidden_dim, num_classes, heads=args.heads, dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights); opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train
    best_val_f1=-1; best_state=None; best_epoch=0; patience_left=args.patience; history=[]
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
        if total_nodes==0: continue
        avg_loss = total_loss/total_nodes
        tr_acc,tr_f1,_ = eval_loader(train_loader)
        va_acc,va_f1,_ = eval_loader(val_loader)
        te_acc,te_f1,_ = eval_loader(test_loader)
        history.append({"epoch":epoch,"loss":avg_loss,"train_acc":tr_acc,"train_f1":tr_f1,"val_acc":va_acc,"val_f1":va_f1,"test_acc":te_acc,"test_f1":te_f1})
        if va_f1>best_val_f1:
            best_val_f1=va_f1; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; patience_left=args.patience
        else:
            patience_left-=1
            if patience_left<=0:
                print(f"Early stopping at epoch {epoch}."); break
        if epoch%10==0 or epoch==1:
            print(f"[{epoch:03d}] loss={avg_loss:.4f} | tr_f1={tr_f1:.3f} va_f1={va_f1:.3f} te_f1={te_f1:.3f}")

    if best_state is not None: model.load_state_dict(best_state)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim, "hidden_dim": args.hidden_dim, "num_classes": num_classes,
        "heads": args.heads, "dropout": args.dropout, "standardize": bool(args.standardize),
        "meta": {"best_val_f1": float(best_val_f1), "best_epoch": int(best_epoch)}
    }, out_dir / "best_model.pt")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print("Done. Best val F1={:.3f} (epoch {}). Out: {}".format(best_val_f1, best_epoch, out_dir))

if __name__ == "__main__":
    main()
