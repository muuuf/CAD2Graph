
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, csv, io, json, argparse
from collections import OrderedDict
import numpy as np

try:
    import networkx as nx
except Exception as e:
    nx = None

PREFERRED_READS = ["utf-8-sig","utf-8","gbk","cp936","latin1"]

def read_csv_guess(path, encs=None):
    encs = encs or PREFERRED_READS
    last_err = None
    for enc in encs:
        try:
            with io.open(path, 'r', encoding=enc, newline='') as f:
                sample = f.read(8192)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except Exception:
                    dialect = csv.excel
                f.seek(0)
                rdr = csv.reader(f, dialect)
                rows = list(rdr)
            return rows, enc
        except Exception as e:
            last_err = e
    raise RuntimeError("Failed to read {} with encodings {}: {}".format(path, encs, last_err))

def coerce_float(x):
    try:
        return float(x)
    except:
        return None

def _normalize_pos(cx_list, cy_list, mode):
    cx = np.array([c if c is not None else np.nan for c in cx_list], dtype=float)
    cy = np.array([c if c is not None else np.nan for c in cy_list], dtype=float)
    if mode == "none":
        return cx, cy
    if mode == "minmax":
        def mm(v):
            m = np.nanmin(v); M = np.nanmax(v)
            if not np.isfinite(m) or not np.isfinite(M) or (M - m) <= 1e-12:
                return v*0.0
            return (v - m) / (M - m)
        return mm(cx), mm(cy)
    # standardize
    def st(v):
        mu = np.nanmean(v); sd = np.nanstd(v)
        if not np.isfinite(sd) or sd <= 1e-12: return v*0.0
        return (v - mu) / sd
    return st(cx), st(cy)

def build_graph_with_labelmap(nodes_rows, edges_rows, label_map,
                              xcol=4, ycol=5, idcol=0, labelcol=1,
                              assume_last_k_feats=10, undirected=True,
                              pos_as_feats=True, pos_norm="standardize", pos_weight=1.0):
    if not nodes_rows: raise RuntimeError("Empty nodes CSV")
    ncols = len(nodes_rows[0])
    k = assume_last_k_feats if assume_last_k_feats is not None else 10
    feat_start = max(0, ncols - k)
    feat_idxs = list(range(feat_start, ncols))

    ids, labels_raw, feats, cx_list, cy_list = [], [], [], [], []
    for r in nodes_rows:
        if not r: continue
        nid = str(r[idcol]) if idcol is not None and idcol < len(r) else None
        if not nid: continue
        ids.append(nid)
        lab = r[labelcol] if (labelcol is not None and labelcol < len(r)) else ""
        lab = "" if lab is None else str(lab)
        labels_raw.append(lab)

        vec = []
        for j in feat_idxs:
            val = coerce_float(r[j]) if j < len(r) else None
            vec.append(0.0 if val is None else float(val))
        feats.append(vec)

        cx = coerce_float(r[xcol]) if (xcol is not None and xcol < len(r)) else None
        cy = coerce_float(r[ycol]) if (ycol is not None and ycol < len(r)) else None
        cx_list.append(cx); cy_list.append(cy)

    # position normalization & append
    if pos_as_feats:
        cxn, cyn = _normalize_pos(cx_list, cy_list, pos_norm)
        for i in range(len(feats)):
            px = 0.0 if not np.isfinite(cxn[i]) else float(cxn[i])
            py = 0.0 if not np.isfinite(cyn[i]) else float(cyn[i])
            feats[i].extend([pos_weight*px, pos_weight*py])

    ids = np.array(ids, dtype=object)
    X = np.array(feats, dtype=float) if feats else np.zeros((0,0), dtype=float)

    # map labels using global label_map
    y = np.array([label_map.get(str(lab), -1) for lab in labels_raw], dtype=np.int64)

    id2idx = {ids[i]: i for i in range(len(ids))}
    G = nx.Graph() if (undirected and nx is not None) else (nx.DiGraph() if nx is not None else None)
    if G is None: raise RuntimeError("networkx is required")

    for i, nid in enumerate(ids):
        G.add_node(nid, idx=i, y=int(y[i]), label=str(labels_raw[i]),
                   x=X[i].tolist(), cx=cx_list[i], cy=cy_list[i])

    missing_ends = 0
    for r in edges_rows:
        if not r or len(r) < 2: continue
        s, t = str(r[0]), str(r[1])
        if s not in id2idx or t not in id2idx:
            missing_ends += 1; continue
        w = 1.0
        if len(r) >= 3:
            ww = coerce_float(r[2])
            if ww is not None: w = ww
        if undirected and t < s: s, t = t, s
        G.add_edge(s, t, weight=float(w))

    info = {"num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "num_missing_edge_endpoints": int(missing_ends),
            "feature_dim": int(X.shape[1]),
            "labels": list(label_map.keys())}
    return G, X, y, ids, info

def save_graphml_without_lists(G, out_path, feature_attr="x"):
    try:
        H = nx.Graph() if not G.is_directed() else nx.DiGraph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges(data=True))
        for n, data in G.nodes(data=True):
            d2 = {}
            for k, v in data.items():
                if k == feature_attr and isinstance(v, (list, tuple)):
                    for i, val in enumerate(v):
                        try:
                            d2["{}{}".format(feature_attr, i)] = float(val)
                        except:
                            d2["{}{}".format(feature_attr, i)] = 0.0
                else:
                    if isinstance(v, (int, float, str)) or v is None:
                        d2[k] = v if v is not None else ""
                    else:
                        d2[k] = str(v)
            H.nodes[n].update(d2)
        nx.write_graphml(H, out_path)
    except Exception as e:
        print("[WARN] GraphML export failed:", e)

def save_outputs(G, X, y, ids, label_map, out_prefix, graphml_mode="explode"):
    if graphml_mode == "explode":
        save_graphml_without_lists(G, out_prefix + ".graphml", feature_attr="x")
    elif graphml_mode == "plain":
        try:
            nx.write_graphml(G, out_prefix + ".graphml")
        except Exception as e:
            print("[WARN] GraphML:", e)
    # edge list
    try:
        with open(out_prefix + ".edgelist","w",encoding="utf-8") as f:
            for u,v,d in G.edges(data=True):
                f.write(f"{u} {v} {d.get('weight',1.0)}\n")
    except Exception as e:
        print("[WARN] edgelist:", e)
    # npz
    try:
        np.savez_compressed(out_prefix + ".npz", x=X, y=y, node_ids=ids)
    except Exception as e:
        print("[WARN] npz:", e)
    # per-graph label map (same global map)
    try:
        with open(out_prefix + "_label_map.json","w",encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[WARN] label_map:", e)
    # pyg
    try:
        import torch
        id2i = {nid:i for i,nid in enumerate(ids)}
        edges = []
        for u,v in G.edges():
            edges.append([id2i[u], id2i[v]])
            if not G.is_directed(): edges.append([id2i[v], id2i[u]])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data = {"x": torch.tensor(X, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.long),
                "edge_index": edge_index}
        torch.save(data, out_prefix + "_pyg.pt")
    except Exception as e:
        print("[INFO] skip PyG:", e)

def main():
    ap = argparse.ArgumentParser(description="Batch-build graphs with a GLOBAL consistent label mapping.")
    ap.add_argument("folder", help="Folder containing CSV files")
    ap.add_argument("--out", required=True, help="Output folder for graphs")
    ap.add_argument("--nodes-suffix", default="_nodes.csv")
    ap.add_argument("--edges-suffix", default="_edges.csv")
    ap.add_argument("--xcol", type=int, default=4)
    ap.add_argument("--ycol", type=int, default=5)
    ap.add_argument("--idcol", type=int, default=0)
    ap.add_argument("--labelcol", type=int, default=1)
    ap.add_argument("--assume-last-k-feats", type=int, default=10)
    ap.add_argument("--directed", action="store_true")
    # position as features
    ap.add_argument("--pos-as-feats", action="store_true", default=True)
    ap.add_argument("--pos-norm", choices=["none","standardize","minmax"], default="standardize")
    ap.add_argument("--pos-weight", type=float, default=1.0)
    # global labels
    ap.add_argument("--classes", help="Comma-separated class names to define the global order, e.g. 交通空间,辅助空间,......")
    ap.add_argument("--label-order", choices=["lex","encounter"], default="lex",
                    help="If --classes not provided, order discovered labels lexicographically or by first encounter (default lex)")
    ap.add_argument("--save-global-map", action="store_true", help="Also save global label_map.json at the output root")
    # graphml
    ap.add_argument("--graphml-mode", choices=["explode","plain","none"], default="explode")
    args = ap.parse_args()

    folder = os.path.normpath(args.folder)
    out_dir = os.path.normpath(args.out)
    if not os.path.isdir(folder):
        print("[ERR] folder is not a directory:", folder); sys.exit(2)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # collect pairs
    nodes_map = {}; edges_map = {}
    for name in os.listdir(folder):
        low = name.lower()
        if low.endswith(args.nodes_suffix.lower()):
            base = name[:-len(args.nodes_suffix)]
            nodes_map[base] = os.path.join(folder, name)
        if low.endswith(args.edges_suffix.lower()):
            base = name[:-len(args.edges_suffix)]
            edges_map[base] = os.path.join(folder, name)
    bases = sorted(set(nodes_map) & set(edges_map))
    if not bases:
        print("[ERR] No matched pairs"); sys.exit(3)

    # -------- First pass: build GLOBAL label map --------
    if args.classes:
        class_list = [c.strip() for c in args.classes.split(",") if c.strip()!=""]
        label_map = OrderedDict((c, i) for i, c in enumerate(class_list))
        print("[INFO] Using user-defined classes (order):", list(label_map.keys()))
    else:
        # discover
        discovered = OrderedDict()
        if args.label_order == "encounter":
            for base in bases:
                rows,_ = read_csv_guess(nodes_map[base])
                for r in rows:
                    if not r: continue
                    lab = r[args.labelcol] if (args.labelcol < len(r)) else ""
                    lab = "" if lab is None else str(lab)
                    if lab not in discovered:
                        discovered[lab] = None
        else:
            # lex: collect then sort
            s = set()
            for base in bases:
                rows,_ = read_csv_guess(nodes_map[base])
                for r in rows:
                    if not r: continue
                    if args.labelcol < len(r):
                        s.add("" if r[args.labelcol] is None else str(r[args.labelcol]))
            for lab in sorted(s):
                discovered[lab] = None
        label_map = OrderedDict((lab, i) for i, lab in enumerate(discovered.keys()))
        print("[INFO] Discovered classes ({}):".format(len(label_map)), list(label_map.keys()))

    if args.save_global_map:
        try:
            with open(os.path.join(out_dir, "global_label_map.json"), "w", encoding="utf-8") as f:
                json.dump(label_map, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print("[WARN] save global label map:", e)

    # -------- Second pass: build each graph with the same mapping --------
    print("[INFO] Building graphs for {} pairs...".format(len(bases)))
    for base in bases:
        try:
            nrows,_ = read_csv_guess(nodes_map[base])
            erows,_ = read_csv_guess(edges_map[base])
            G, X, y, ids, info = build_graph_with_labelmap(
                nrows, erows, label_map,
                xcol=args.xcol, ycol=args.ycol, idcol=args.idcol, labelcol=args.labelcol,
                assume_last_k_feats=args.assume_last_k_feats, undirected=(not args.directed),
                pos_as_feats=args.pos_as_feats, pos_norm=args.pos_norm, pos_weight=args.pos_weight
            )
            prefix = os.path.join(out_dir, base, base)
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            save_outputs(G, X, y, ids, label_map, prefix, graphml_mode=args.graphml_mode)
            print("[OK]", base, "nodes:", info["num_nodes"], "edges:", info["num_edges"], "feat_dim:", info["feature_dim"])
        except Exception as e:
            print("[ERR]", base, ":", e)

    print("[DONE] All graphs saved to:", out_dir)

if __name__ == "__main__":
    main()
