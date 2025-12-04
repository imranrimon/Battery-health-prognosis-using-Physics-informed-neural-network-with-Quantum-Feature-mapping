# qk/compare_train_time_models.py
import os, re, glob, json, argparse, time
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- settings ----------
LOG_CANDIDATES = (
    "metrics.csv",        # our QK/TJU/HUST scripts
    "logging.txt",        # common
    "log.txt",            # variant
    "train.log",          # variant
    "train_log.txt",      # variant
)

KNOWN_BATCH_NAMES = {
    "2C","3C","R2.5","R3","RW","satellite",    # XJTU tags
    "batch0","batch1","batch2","0","1","2",    # TJU tags
    "XJTU","TJU","HUST","MIT"                  # datasets
}

# more liberal epoch timing patterns
EPOCH_PATTERNS = [
    re.compile(r"epoch[:\s]*(\d+).*?(?:secs?|sec|second|time|cost|耗时)[:=\s]*([0-9]*\.?[0-9]+)\s*s?\b", re.IGNORECASE),
    re.compile(r"epoch[:\s]*(\d+).*?\(\s*([0-9]*\.?[0-9]+)\s*s\s*\)", re.IGNORECASE),
    re.compile(r"\bepoch[:\s]*(\d+)\b.*?took[:=\s]*([0-9]*\.?[0-9]+)\s*s", re.IGNORECASE),
]
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")

# common epoch-time column names in metrics.csv
METRICS_TIME_COLUMNS = ["secs", "sec", "seconds", "time", "epoch_time"]

def _fmt_hms(sec):
    s = float(sec)
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def _infer_batch_tag(model_root, run_path):
    rel = os.path.relpath(run_path, model_root)
    parts = rel.split(os.sep)
    for p in reversed(parts[:-1]):
        if p in KNOWN_BATCH_NAMES:
            return p
    for p in reversed(parts[:-1]):
        low = p.lower()
        if any(k in low for k in ["xjtu","tju","hust","mit","experiment","exp"]):
            return p
    return parts[-2] if len(parts) >= 2 else "batch"

def _find_runs_recursively(dataset_root):
    run_dirs = set()
    for name in LOG_CANDIDATES:
        for f in glob.glob(os.path.join(dataset_root, "**", name), recursive=True):
            run_dirs.add(os.path.dirname(f))
    # also accept folders that have summary/best files (fallback timing via timestamps)
    for f in glob.glob(os.path.join(dataset_root, "**", "summary.json"), recursive=True):
        run_dirs.add(os.path.dirname(f))
    for f in glob.glob(os.path.join(dataset_root, "**", "best_metrics.json"), recursive=True):
        run_dirs.add(os.path.dirname(f))

    batches = {}
    for r in sorted(run_dirs):
        b = _infer_batch_tag(dataset_root, r)
        batches.setdefault(b, []).append(r)
    return batches

def _best_epoch_from_df_or_file(run_path, df):
    bm = os.path.join(run_path, "best_metrics.json")
    if os.path.exists(bm):
        try:
            with open(bm, "r", encoding="utf-8") as f:
                k = json.load(f)
            be = int(k.get("best_epoch"))
            if be >= 1:
                return be
        except Exception:
            pass
    if df is not None and "val_total" in df.columns and len(df) > 0:
        idx = int(df["val_total"].idxmin())
        return int(df.iloc[idx]["epoch"])
    if df is not None and "epoch" in df.columns and len(df) > 0:
        return int(df["epoch"].max())
    return 0

def _epoch_seconds_from_metrics_csv(run_path):
    mpath = os.path.join(run_path, "metrics.csv")
    if not os.path.exists(mpath):
        return None
    try:
        df = pd.read_csv(mpath)
    except Exception:
        return None
    # find a usable time column
    time_col = None
    for c in METRICS_TIME_COLUMNS:
        if c in df.columns and not df[c].dropna().empty:
            time_col = c
            break
    if time_col is None:
        return None
    df = df.dropna(subset=[time_col])
    if len(df) == 0:
        return None

    epochs = int(df["epoch"].max()) if "epoch" in df.columns else len(df)
    total_sec = float(df[time_col].sum())
    mean_sec  = float(df[time_col].mean())
    med_sec   = float(df[time_col].median())
    std_sec   = float(df[time_col].std(ddof=0)) if len(df) > 1 else 0.0
    best_ep   = _best_epoch_from_df_or_file(run_path, df)
    if best_ep >= 1 and "epoch" in df.columns:
        ttb_sec = float(df.loc[df["epoch"] <= best_ep, time_col].sum())
    else:
        ttb_sec = total_sec

    return {
        "source": f"metrics.csv[{time_col}]",
        "run": os.path.basename(run_path),
        "epochs": epochs,
        "best_epoch": best_ep if best_ep >= 1 else epochs,
        "total_sec": total_sec,
        "avg_epoch_sec": mean_sec,
        "median_epoch_sec": med_sec,
        "std_epoch_sec": std_sec,
        "time_to_best_sec": ttb_sec,
    }

def _parse_timestamps(lines):
    ts = []
    for ln in lines:
        m = TS_RE.search(ln)
        if not m:
            ts.append(None)
            continue
        dt = datetime.strptime(m.group(1)+" "+m.group(2), "%Y-%m-%d %H:%M:%S")
        ts.append(dt)
    return ts

def _epoch_seconds_from_logs(run_path):
    log_path = None
    for cand in LOG_CANDIDATES[1:]:  # skip metrics.csv
        p = os.path.join(run_path, cand)
        if os.path.exists(p):
            log_path = p; break
    if log_path is None:
        return None

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None
    if not lines:
        return None

    epoch_secs = {}
    cur_epoch = None

    for ln in lines:
        for pat in EPOCH_PATTERNS:
            m = pat.search(ln)
            if m:
                ep = int(m.group(1))
                sec = float(m.group(2))
                epoch_secs[ep] = sec
                cur_epoch = ep
                break
        else:
            m_ep = re.search(r"epoch[:\s]*(\d+)", ln, re.IGNORECASE)
            if m_ep:
                cur_epoch = int(m_ep.group(1))
            m_took = re.search(r"(?:took|耗时)[:=\s]*([0-9]*\.?[0-9]+)\s*s", ln, re.IGNORECASE)
            if m_took and cur_epoch is not None and cur_epoch not in epoch_secs:
                epoch_secs[cur_epoch] = float(m_took.group(1))

    if not epoch_secs:
        return None

    epochs = int(max(epoch_secs))
    secs_series = [epoch_secs.get(e, np.nan) for e in range(1, epochs+1)]
    secs_series = [s for s in secs_series if pd.notna(s)]
    if not secs_series:
        return None

    total_sec = float(np.nansum(secs_series))
    mean_sec  = float(np.nanmean(secs_series))
    med_sec   = float(np.nanmedian(secs_series))
    std_sec   = float(np.nanstd(secs_series))
    best_ep   = _best_epoch_from_df_or_file(run_path, None)
    if best_ep <= 0:
        best_ep = epochs
    ttb_sec   = float(np.nansum([epoch_secs.get(e, 0.0) for e in range(1, best_ep+1)]))

    return {
        "source": os.path.basename(log_path),
        "run": os.path.basename(run_path),
        "epochs": epochs,
        "best_epoch": best_ep,
        "total_sec": total_sec,
        "avg_epoch_sec": mean_sec,
        "median_epoch_sec": med_sec,
        "std_epoch_sec": std_sec,
        "time_to_best_sec": ttb_sec,
    }

def _ctime(path):
    try:
        return os.path.getctime(path)
    except Exception:
        return None

def _mtime(path):
    try:
        return os.path.getmtime(path)
    except Exception:
        return None

def _estimate_from_timestamps(run_path):
    """
    Fallback when no per-epoch timing is available:
    use creation time of earliest log/metrics/checkpoint vs latest modified time
    of common artifacts (metrics, logs, best/last checkpoints, preds).
    """
    candidates = []
    for nm in ("metrics.csv","logging.txt","log.txt","train.log","train_log.txt",
               "summary.json","best_metrics.json","best_qk.pt","last_qk.pt",
               "preds_valid.csv","preds_test.csv"):
        p = os.path.join(run_path, nm)
        if os.path.exists(p):
            candidates.append(p)
    if not candidates:
        return None

    ctimes = [t for t in (_ctime(p) for p in candidates) if t is not None]
    mtimes = [t for t in (_mtime(p) for p in candidates) if t is not None]
    if not ctimes or not mtimes:
        return None

    total_sec = max(mtimes) - min(ctimes)
    if total_sec <= 0:
        total_sec = max(1.0, total_sec)

    # we cannot know epochs precisely; set to NaNs except total
    return {
        "source": "timestamp_fallback",
        "run": os.path.basename(run_path),
        "epochs": np.nan,
        "best_epoch": np.nan,
        "total_sec": float(total_sec),
        "avg_epoch_sec": np.nan,
        "median_epoch_sec": np.nan,
        "std_epoch_sec": np.nan,
        "time_to_best_sec": np.nan,
    }

def _read_run_time_stats(run_path):
    return (
        _epoch_seconds_from_metrics_csv(run_path)
        or _epoch_seconds_from_logs(run_path)
        or _estimate_from_timestamps(run_path)
    )

def summarize_model(model_name, dataset_root, verbose=True):
    batches = _find_runs_recursively(dataset_root)
    rows, batch_avgs = [], []

    if verbose:
        tot_runs = sum(len(v) for v in batches.values())
        print(f"[{model_name}] Found {tot_runs} run(s) under {dataset_root}")
        for b, lst in batches.items():
            print(f"  - {b}: {len(lst)} run(s)")

    for batch_tag, run_paths in batches.items():
        run_rows = []
        for rp in run_paths:
            s = _read_run_time_stats(rp)
            if s is None:
                if verbose:
                    print(f"    ! Skipping (no metrics/log parse): {rp}")
                continue
            s["model"] = model_name
            s["batch"] = batch_tag
            run_rows.append(s)
            rows.append(s)

        if run_rows:
            rdf = pd.DataFrame(run_rows)
            num_cols = ["epochs","best_epoch","total_sec","avg_epoch_sec","median_epoch_sec","std_epoch_sec","time_to_best_sec"]
            means = rdf[num_cols].mean(numeric_only=True)
            avg_row = {"model": model_name, "batch": batch_tag, "run": "AVG_over_runs"}
            for k, v in means.items():
                avg_row[k] = float(v) if pd.notna(v) else np.nan
            batch_avgs.append(avg_row)

    per_run_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["model","batch","run","epochs","best_epoch","total_sec","avg_epoch_sec","median_epoch_sec","std_epoch_sec","time_to_best_sec","source"]
    )
    per_batch_avg_df = pd.DataFrame(batch_avgs) if batch_avgs else pd.DataFrame(
        columns=["model","batch","run","epochs","best_epoch","total_sec","avg_epoch_sec","median_epoch_sec","std_epoch_sec","time_to_best_sec"]
    )
    return per_run_df, per_batch_avg_df

def compare_models(models, output_root, verbose=True):
    os.makedirs(output_root, exist_ok=True)
    all_runs, all_batch_avg = [], []

    for label, root in models:
        pr, pb = summarize_model(label, root, verbose=verbose)
        all_runs.append(pr)
        all_batch_avg.append(pb)

    per_run_df = pd.concat(all_runs, ignore_index=True) if all_runs else pd.DataFrame()
    per_batch_avg_df = pd.concat(all_batch_avg, ignore_index=True) if all_batch_avg else pd.DataFrame()

    for df in (per_run_df, per_batch_avg_df):
        if not df.empty and "total_sec" in df.columns:
            df["total_hms"]        = df["total_sec"].apply(lambda x: _fmt_hms(x) if pd.notna(x) else "")
            if "avg_epoch_sec" in df.columns:
                df["avg_epoch_hms"]    = df["avg_epoch_sec"].apply(lambda x: _fmt_hms(x) if pd.notna(x) else "")
            if "median_epoch_sec" in df.columns:
                df["median_epoch_hms"] = df["median_epoch_sec"].apply(lambda x: _fmt_hms(x) if pd.notna(x) else "")
            if "time_to_best_sec" in df.columns:
                df["time_to_best_hms"] = df["time_to_best_sec"].apply(lambda x: _fmt_hms(x) if pd.notna(x) else "")

    overall_rows = []
    if not per_batch_avg_df.empty:
        for model_name, g in per_batch_avg_df.groupby("model"):
            num_cols = ["epochs","best_epoch","total_sec","avg_epoch_sec","median_epoch_sec","std_epoch_sec","time_to_best_sec"]
            means = g[num_cols].mean(numeric_only=True)
            row = {"model": model_name}
            for k, v in means.items(): row[k] = float(v) if pd.notna(v) else np.nan
            row["total_hms"]        = _fmt_hms(row["total_sec"]) if pd.notna(row["total_sec"]) else ""
            row["avg_epoch_hms"]    = _fmt_hms(row["avg_epoch_sec"]) if pd.notna(row["avg_epoch_sec"]) else ""
            row["median_epoch_hms"] = _fmt_hms(row["median_epoch_sec"]) if pd.notna(row["median_epoch_sec"]) else ""
            row["time_to_best_hms"] = _fmt_hms(row["time_to_best_sec"]) if pd.notna(row["time_to_best_sec"]) else ""
            overall_rows.append(row)
    overall_by_model_df = pd.DataFrame(overall_rows)

    per_run_csv   = os.path.join(output_root, "per_model_per_run.csv")
    per_batch_csv = os.path.join(output_root, "per_model_per_batch_avg.csv")
    overall_csv   = os.path.join(output_root, "overall_by_model.csv")

    if not per_run_df.empty:          per_run_df.to_csv(per_run_csv, index=False)
    if not per_batch_avg_df.empty:    per_batch_avg_df.to_csv(per_batch_csv, index=False)
    if not overall_by_model_df.empty: overall_by_model_df.to_csv(overall_csv, index=False)

    print("[OK] Wrote:")
    if not per_run_df.empty:          print("  ", per_run_csv)
    if not per_batch_avg_df.empty:    print("  ", per_batch_csv)
    if not overall_by_model_df.empty: print("  ", overall_csv)

def parse_args():
    ap = argparse.ArgumentParser("Compare training time across models (recursive; many fallbacks)")
    ap.add_argument("--model", action="append", required=True,
                    help='Model spec NAME=ROOT (e.g., QK="E:\\path\\results_qk_agg"). Use multiple --model.')
    ap.add_argument("--out", type=str, required=True, help="Output folder for CSVs")
    ap.add_argument("--quiet", action="store_true", help="Reduce console output")
    return ap.parse_args()

def main():
    args = parse_args()
    pairs = []
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"Bad --model spec: {spec}. Use NAME=ROOT")
        name, root = spec.split("=", 1)
        name, root = name.strip(), root.strip().strip('"')
        if not os.path.isdir(root):
            print(f"[WARN] Not a directory: {root}")
        pairs.append((name, root))
    compare_models(pairs, args.out, verbose=not args.quiet)

if __name__ == "__main__":
    main()
