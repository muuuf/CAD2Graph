# -*- coding: utf-8 -*-
"""
Timeout-safe, TOP-ONLY extractor for *visible* TextEntity/TextDot from Rhino .3dm files.
- Recursively scans input folder (case-insensitive .3dm)
- Per-file subprocess with timeout (default: 60s; configurable via --timeout)
- Writes UTF-8 (with BOM) .txt output (tab-separated)
- Ignores block (InstanceDefinition) contents entirely (TOP-ONLY)
- **Only exports text from objects that are visible AND on visible layers**
- Detailed logging and a manifest of discovered files

Usage:
    pip install rhino3dm
    python extract_rhino_texts_top_only_visible_timeout_txt.py --input "D:/path/to/folder" --output "D:/out.txt" --log "D:/out.log" --timeout 60

Output columns (tab-separated):
    file    layer   kind    text
"""
from __future__ import print_function
import os
import sys
import argparse
import datetime
import multiprocessing as mp

# ----------------- helpers ----------------------
def safe_text(s):
    if s is None:
        return ""
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return ""
    s = s.replace("\r\n", " ").replace("\n", " ").replace("\t", " ").strip()
    return s

def enumerate_3dm_files(root_dir):
    """Recursively yield absolute paths of files ending with .3dm (case-insensitive)."""
    for base, _dirs, files in os.walk(root_dir):
        for name in files:
            if name.lower().endswith(".3dm"):
                yield os.path.abspath(os.path.join(base, name))

# ------------- child worker imports -------------
def _import_rhino3dm():
    try:
        import rhino3dm  # type: ignore
        return rhino3dm
    except Exception as e:
        sys.stderr.write("Child failed to import rhino3dm: %s\n" % e)
        return None

# ---------------- rhino extract -----------------
def read_text_from_geo(geo):
    """Return (kind, text) if geometry bears text, else (None, None)."""
    typ = type(geo).__name__
    # TextEntity (文字方块)
    if "TextEntity" in typ:
        for attr in ("PlainText", "Text", "RichText"):
            if hasattr(geo, attr):
                try:
                    val = getattr(geo, attr)
                    val = val() if callable(val) else val
                    if val:
                        return ("TextEntity", safe_text(val))
                except Exception:
                    pass
    # TextDot (文字点)
    if "TextDot" in typ:
        for attr in ("Text", "PlainText", "RichText"):
            if hasattr(geo, attr):
                try:
                    val = getattr(geo, attr)
                    val = val() if callable(val) else val
                    if val:
                        return ("TextDot", safe_text(val))
                except Exception:
                    pass
    # Fallback: any object exposing PlainText/RichText/Text
    for attr in ("PlainText", "RichText", "Text"):
        if hasattr(geo, attr):
            try:
                val = getattr(geo, attr)
                val = val() if callable(val) else val
                if val:
                    return (typ, safe_text(val))
            except Exception:
                pass
    return (None, None)

def build_layer_map_and_vis(model):
    """
    Returns:
        layer_map: {index -> name}
        layer_vis: {index -> bool_visible}
    """
    name_map = {}
    vis_map = {}
    try:
        for i in range(len(model.Layers)):
            lyr = model.Layers[i]
            # name
            try:
                name_map[i] = getattr(lyr, "Name", "")
            except Exception:
                name_map[i] = ""
            # visibility (兼容不同 rhino3dm 版本：IsVisible / Visible)
            vis = True
            for attr in ("IsVisible", "Visible"):
                if hasattr(lyr, attr):
                    try:
                        v = getattr(lyr, attr)
                        vis = bool(v() if callable(v) else v)
                        break
                    except Exception:
                        continue
            vis_map[i] = vis
    except Exception:
        pass
    return name_map, vis_map

def obj_is_visible(attrs):
    """
    Best-effort object-level visibility:
      - prefer Attributes.Visible / IsVisible if present
      - fall back to True
    """
    if attrs is None:
        return True
    for attr in ("Visible", "IsVisible"):
        if hasattr(attrs, attr):
            try:
                v = getattr(attrs, attr)
                return bool(v() if callable(v) else v)
            except Exception:
                pass
    return True

def _child_extract_top_only_visible(file_path, queue):
    """Run inside a subprocess: read one file and push lines back via Queue (TOP-ONLY, VISIBLE ONLY)."""
    rhino3dm = _import_rhino3dm()
    if rhino3dm is None:
        queue.put(("error", "import", "Failed to import rhino3dm"))
        return

    try:
        model = rhino3dm.File3dm.Read(file_path)
        if model is None:
            queue.put(("error", "read", "File3dm.Read returned None"))
            return
    except Exception as e:
        queue.put(("error", "read", str(e)))
        return

    layer_map, layer_vis = build_layer_map_and_vis(model)

    out_lines = []
    base = os.path.basename(file_path)

    # TOP-ONLY: only iterate top-level model.Objects, no blocks handling
    for obj in model.Objects:
        try:
            attrs = getattr(obj, "Attributes", None)
            # 1) object visibility
            if not obj_is_visible(attrs):
                continue

            # 2) layer visibility
            layer_name = ""
            layer_visible = True
            try:
                layer_idx = attrs.LayerIndex if attrs else None
                if layer_idx is not None:
                    layer_name = layer_map.get(layer_idx, "")
                    layer_visible = layer_vis.get(layer_idx, True)
            except Exception:
                pass
            if not layer_visible:
                continue

            # 3) geometry text
            geo = getattr(obj, "Geometry", None)
            if geo is None:
                continue
            kind, text = read_text_from_geo(geo)
            if not kind or not text:
                continue

            out_lines.append("{}\t{}\t{}\t{}\n".format(base, safe_text(layer_name), kind, text))
        except Exception:
            continue

    queue.put(("ok", out_lines))

# -------------------- main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Timeout-safe TOP-ONLY *visible* texts extractor for Rhino .3dm to TXT")
    ap.add_argument("--input", "-i", required=True, help="Root folder to scan (recursively)")
    ap.add_argument("--output", "-o", required=True, help="Output .txt path (UTF-8 with BOM)")
    ap.add_argument("--log", "-l", default="", help="Optional log file path")
    ap.add_argument("--timeout", "-t", type=int, default=60, help="Per-file timeout in seconds (default: 60)")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)
    log_path = os.path.abspath(args.log) if args.log else ""

    if not os.path.isdir(in_dir):
        print("ERROR: Input folder does not exist: {}".format(in_dir), file=sys.stderr)
        sys.exit(2)

    # Gather files
    files = list(enumerate_3dm_files(in_dir))
    total_files = len(files)
    if total_files == 0:
        print("WARN: No .3dm files found under {}".format(in_dir), file=sys.stderr)

    # Prepare output & log
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    log_fh = None
    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            log_fh = open(log_path, "w", encoding="utf-8")
        except Exception as e:
            print("WARN: Cannot open log file '{}': {} (fallback to stderr)".format(log_path, e), file=sys.stderr)

    # Write manifest for diagnostics
    manifest_path = out_path + ".files.txt"
    with open(manifest_path, "w", encoding="utf-8-sig", newline="") as mf:
        for p in files:
            mf.write(p + "\n")

    # Process each file in a separate process with timeout
    processed = 0
    total_items = 0
    with open(out_path, "w", encoding="utf-8-sig", newline="") as out_fh:
        out_fh.write("file\tlayer\tkind\ttext\n")
        for p in files:
            processed += 1
            q = mp.Queue()
            proc = mp.Process(target=_child_extract_top_only_visible, args=(p, q))
            proc.start()
            proc.join(args.timeout)

            if proc.is_alive():
                # Timeout: kill and log
                proc.terminate()
                proc.join()
                msg = "[{}] TIMEOUT after {}s: {}\n".format(datetime.datetime.now(), args.timeout, p)
                (log_fh.write(msg) if log_fh else sys.stderr.write(msg))
                print("({}/{}) TIMEOUT: {}".format(processed, total_files, os.path.basename(p)))
                continue

            # Read result from queue
            try:
                status, payload, *rest = q.get_nowait()
            except Exception:
                status, payload = "error", "no_result"

            if status == "ok":
                lines = payload
                for line in lines:
                    out_fh.write(line)
                total_items += len(lines)
                print("({}/{}) OK items:+{} total:{} -> {}".format(processed, total_files, len(lines), total_items, os.path.basename(p)))
            else:
                reason = payload
                msg = "[{}] ERROR {}: {}\n".format(datetime.datetime.now(), reason, p)
                (log_fh.write(msg) if log_fh else sys.stderr.write(msg))
                print("({}/{}) ERROR ({}) -> {}".format(processed, total_files, reason, os.path.basename(p)))

    if log_fh:
        log_fh.close()

    print("\nDone. Files scanned: {} | Text items: {}".format(total_files, total_items))
    print("Output written to: {}".format(out_path))
    print("Manifest written to: {}".format(manifest_path))
    if log_path:
        print("Errors/timeouts (if any) logged to: {}".format(log_path))

if __name__ == "__main__":
    # On Windows, freeze support for multiprocessing
    mp.freeze_support()
    main()
