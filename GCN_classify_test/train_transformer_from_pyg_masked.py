
# -*- coding: utf-8 -*-
"""
Masked-label node classification with TransformerConv.
- For each graph, randomly mask a fraction of node labels.
- Train: compute loss ONLY on masked nodes (unmasked labels are "given"; not input features).
- Val/Test: also apply masking; metrics can be evaluated on masked-only or all nodes (configurable).

Usage:
  python train_transformer_from_pyg_masked.py --input_dir C:/graphs_out --out_dir ./runs/masked ^
    --label-mask-train 0.5 --label-mask-val 0.5 --label-mask-test 0.5 --eval-on masked
"""
import argparse, json, os, random, math, csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv

def set_seed(seed:int):
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
        'y': y.long(),
        'node_ids': g.get('node_ids', [str(i) for i in range(g['x'].shape[0])]),
        'meta': g.get('meta', {})
    }
    out['meta']['key'] = out['meta'].get('key', p.stem)
    return out

def to_pyg_data(g: Dict) -> "Data":
    data = Data(x=g['x'], edge_index=g['edge_index'], y=g['y'])
    data.node_ids = g['node_ids']; data.key = g['meta']['key']
    return data

class GraphTransformer2Layer(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, heads: int=8, dropout: float=0.2):
        super().__init__()
        self.conv1 = TransformerConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = TransformerConv(hidden * heads, num_classes, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

def make_mask(n_nodes:int, frac:float, rng:random.Random) -> torch.Tensor:
    frac = max(0.0, min(1.0, float(frac)))
    k = int(round(frac * n_nodes))
    idx = list(range(n_nodes)); rng.shuffle(idx)
    masked_idx = set(idx[:k])
    m = torch.zeros(n_nodes, dtype=torch.bool)
    if k>0: m[list(masked_idx)] = True
    return m

def apply_masks_per_split(graphs: list, frac: float, seed: int) -> list:
    out = []
    for i, g in enumerate(graphs):
        rng = random.Random((seed+1)*100003 + i*97)
        m = make_mask(g['x'].shape[0], frac, rng)
        g = dict(g); g['mask'] = m; out.append(g)
    return out

@torch.no_grad()
def evaluate(model, loader, device, num_classes, eval_on:str="masked") -> Tuple[float, float]:
    model.eval()
    all_pred = []; all_true = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        pred = logits.argmax(dim=-1)
        y = batch.y
        if eval_on == "masked" and hasattr(batch, "mask"):
            scope = batch.mask.bool()
        else:
            scope = torch.ones_like(y, dtype=torch.bool)
        scope = scope & (y >= 0)
        if scope.sum() == 0:
            continue
        all_pred.append(pred[scope].cpu())
        all_true.append(y[scope].cpu())
    if not all_pred:
        return float('nan'), float('nan')
    pred_all = torch.cat(all_pred, dim=0); true_all = torch.cat(all_true, dim=0)
    acc = accuracy(pred_all, true_all); f1 = f1_macro(pred_all, true_all, num_classes)
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir",   required=True)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio",   type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--standardize", action="store_true", default=True)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_standardize", action="store_false", dest="standardize")
    parser.add_argument("--no_class_weights", action="store_false", dest="use_class_weights")
    parser.add_argument("--label-mask-train", type=float, default=0.5)
    parser.add_argument("--label-mask-val",   type=float, default=0.5)
    parser.add_argument("--label-mask-test",  type=float, default=0.5)
    parser.add_argument("--mask-seed", type=int, default=2025)
    parser.add_argument("--eval-on", choices=["masked","all"], default="masked")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dir = Path(args.input_dir); out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pt_files = sorted(list(input_dir.rglob("*_pyg.pt"))) or sorted(list(input_dir.rglob("*.pt")))
    if not pt_files: raise RuntimeError(f"No .pt graphs found under {input_dir}")

    all_graphs = [robust_load_graph_pt(p) for p in pt_files]
    random.shuffle(all_graphs)

    n = len(all_graphs); n_tr = int(n*args.train_ratio); n_va = int(n*args.val_ratio)
    Gtr = all_graphs[:n_tr]; Gva = all_graphs[n_tr:n_tr+n_va]; Gte = all_graphs[n_tr+n_va:]
    print(f"Dataset split: {len(Gtr)} train, {len(Gva)} val, {len(Gte)} test")

    mean = torch.zeros(1); std = torch.ones(1)
    if args.standardize:
        Xtr = torch.cat([g['x'] for g in Gtr if g['x'].shape[0]>0], dim=0)
        if Xtr.shape[0]>0: _, mean, std = zscore(Xtr)
        for g in all_graphs:
            if g['x'].shape[0]>0: g['x'], _, _ = zscore(g['x'].float(), mean, std)

    Gtr = apply_masks_per_split(Gtr, args.label_mask_train, args.mask_seed)
    Gva = apply_masks_per_split(Gva, args.label_mask_val,   args.mask_seed+1)
    Gte = apply_masks_per_split(Gte, args.label_mask_test,  args.mask_seed+2)

    def to_data_list(graphs):
        lst = []
        for g in graphs:
            d = to_pyg_data(g); d.mask = g['mask']; lst.append(d)
        return lst

    train_dataset = to_data_list(Gtr); val_dataset = to_data_list(Gva); test_dataset = to_data_list(Gte)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    in_dim = int(train_dataset[0].num_node_features)
    all_y  = torch.cat([d.y for d in train_dataset + val_dataset + test_dataset if d.y.numel()>0])
    num_classes = int(all_y.max().item()) + 1
    print(f"in_dim={in_dim}, num_classes={num_classes}, eval_on={args.eval_on}")

    class_weights = None
    if args.use_class_weights:
        ytr = torch.cat([d.y for d in train_dataset if d.y.numel()>0])
        cnt = torch.bincount(ytr, minlength=num_classes).float()
        w = 1.0 / torch.clamp(cnt, min=1.0); w = w / w.mean()
        class_weights = w.to(device)

    model = GraphTransformer2Layer(in_dim, args.hidden_dim, num_classes, heads=args.heads, dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_f1=-1.0; best_state=None; best_epoch=0; patience_left=args.patience; history=[]
    for epoch in range(1, args.epochs+1):
        model.train(); total_loss=0.0; total_count=0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            scope = batch.mask.bool() & (batch.y >= 0)
            if scope.sum()==0: continue
            loss = loss_fn(logits[scope], batch.y[scope])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item()) * int(scope.sum().item())
            total_count += int(scope.sum().item())
        avg_loss = (total_loss/total_count) if total_count>0 else float('nan')

        def eval_loader(loader):
            return evaluate(model, loader, device, num_classes, eval_on=args.eval_on)
        tr_acc,tr_f1 = eval_loader(train_loader)
        va_acc,va_f1 = eval_loader(val_loader)
        te_acc,te_f1 = eval_loader(test_loader)
        history.append({"epoch":epoch,"loss":avg_loss,"train_acc":tr_acc,"train_f1":tr_f1,"val_acc":va_acc,"val_f1":va_f1,"test_acc":te_acc,"test_f1":te_f1})
        if va_f1>best_val_f1:
            best_val_f1=va_f1; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; patience_left=args.patience
        else:
            patience_left-=1
            if patience_left<=0:
                print(f"Early stopping at epoch {epoch}."); break
        if epoch%10==0 or epoch==1:
            print(f"[Epoch {epoch:03d}] loss={avg_loss:.4f} | train_f1={tr_f1:.3f} val_f1={va_f1:.3f} test_f1={te_f1:.3f}")

    if best_state is not None: model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim, "hidden_dim": args.hidden_dim, "num_classes": num_classes,
        "heads": args.heads, "dropout": args.dropout, "standardize": bool(args.standardize),
        "masking": {"train": float(args.label_mask_train), "val": float(args.label_mask_val), "test": float(args.label_mask_test),
                    "eval_on": args.eval_on, "mask_seed": int(args.mask_seed)},
        "meta": {"best_val_f1": float(best_val_f1), "best_epoch": int(best_epoch)}
    }, out_dir / "best_model.pt")
    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"Done. Best val F1={best_val_f1:.3f} (epoch {best_epoch}). Out: {out_dir}")

if __name__ == "__main__":
    main()
