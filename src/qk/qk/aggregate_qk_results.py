import os, json, glob, math, argparse
import numpy as np
import pandas as pd

# ---------- metrics ----------
def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred)**2))

def rmse(y_true, y_pred):
    return float(math.sqrt(mse(y_true, y_pred)))

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))

def metrics_from_preds(y_true, y_pred):
    return {
        "MSE":  mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2":   r2(y_true, y_pred),
    }

# For TRAIN when preds_train.csv is absent, fall back to train_L_data (MSE) from metrics.csv
def train_metrics_from_logs(metrics_csv):
    if not os.path.exists(metrics_csv):
        return None
    try:
        df = pd.read_csv(metrics_csv)
        # choose best epoch as the one with minimal val_total if present; else last row
        if "val_total" in df.columns:
            idx = int(df["val_total"].idxmin())
        else:
            idx = len(df) - 1
        row = df.iloc[idx]
        if "train_L_data" in df.columns and not np.isnan(row["train_L_data"]):
            m = float(row["train_L_data"])
            return {"MSE": m, "RMSE": math.sqrt(max(m, 0.0)), "MAPE": np.nan, "R2": np.nan}
    except Exception:
        pass
    return None

# ---------- I/O helpers ----------
def load_preds_csv(path):
    # expected columns: ["t","y_true","y_pred"] (as written by your QK-PINN scripts)
    df = pd.read_csv(path)
    return df["y_true"].values, df["y_pred"].values

def find_runs(dataset_root):
    """
    dataset_root example:
      XJTU: results_qk/XJTU
      TJU : results_qk/TJU/batch0 (or results_qk/TJU/<sub>/...)
      HUST: results_qk/HUST
      MIT : results_qk/MIT

    We assume: dataset_root/<batch_tag>/<run_folder>/...
    Where <run_folder> contains metrics.csv, preds_valid.csv, preds_test.csv (and optionally preds_train.csv)
    """
    batch_dirs = []
    for p in sorted(glob.glob(os.path.join(dataset_root, "*"))):
        if os.path.isdir(p):
            batch_dirs.append(p)

    # If there are no subdirs (rare), treat dataset_root as the batch_dir itself
    if not batch_dirs:
        batch_dirs = [dataset_root]

    discovered = {}
    for bdir in batch_dirs:
        # <run_folder> can be any subdir that has at least metrics.csv or preds_valid.csv/preds_test.csv
        runs = [p for p in glob.glob(os.path.join(bdir, "*")) if os.path.isdir(p)]
        valid_runs = []
        for r in runs:
            files = os.listdir(r)
            if any(fn in files for fn in ("metrics.csv", "preds_valid.csv", "preds_test.csv", "summary.json", "best_metrics.json")):
                valid_runs.append(r)
        if valid_runs:
            discovered[os.path.basename(bdir)] = sorted(valid_runs)
    return discovered  # {batch_tag: [run_paths...]}

def summarize_batch(batch_tag, run_paths, out_dir):
    """
    For a single batch tag, compute per-run metrics + average, for Train/Valid/Test.
    Outputs:
      - CSV with one row per run + average row
      - JSON summary with averages
    """
    rows = []
    for rpath in run_paths:
        run_name = os.path.basename(rpath)

        # Train metrics
        train_csv = os.path.join(rpath, "preds_train.csv")  # optional
        metrics_csv = os.path.join(rpath, "metrics.csv")
        if os.path.exists(train_csv):
            y_tr, yhat_tr = load_preds_csv(train_csv)
            tr = metrics_from_preds(y_tr, yhat_tr)
        else:
            tr = train_metrics_from_logs(metrics_csv)  # may be None or {MSE, RMSE, ...}
        if tr is None:
            tr = {"MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

        # Valid metrics
        val_csv = os.path.join(rpath, "preds_valid.csv")
        if os.path.exists(val_csv):
            y_va, yhat_va = load_preds_csv(val_csv)
            va = metrics_from_preds(y_va, yhat_va)
        else:
            va = {"MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

        # Test metrics
        test_csv = os.path.join(rpath, "preds_test.csv")
        if os.path.exists(test_csv):
            y_te, yhat_te = load_preds_csv(test_csv)
            te = metrics_from_preds(y_te, yhat_te)
        else:
            te = {"MSE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}

        rows.append({
            "batch": batch_tag, "run": run_name,
            "train_MSE": tr["MSE"], "train_RMSE": tr["RMSE"], "train_MAPE": tr["MAPE"], "train_R2": tr["R2"],
            "valid_MSE": va["MSE"], "valid_RMSE": va["RMSE"], "valid_MAPE": va["MAPE"], "valid_R2": va["R2"],
            "test_MSE":  te["MSE"], "test_RMSE":  te["RMSE"],  "test_MAPE":  te["MAPE"],  "test_R2":  te["R2"],
        })

    df = pd.DataFrame(rows)

    # Average across runs (ignore NaNs)
    avg = df.drop(columns=["batch", "run"]).mean(numeric_only=True)
    avg_row = {"batch": batch_tag, "run": "AVG_over_runs"}
    for k, v in avg.items():
        avg_row[k] = float(v)
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    # Save CSV + JSON for this batch
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{batch_tag}_per_run_and_avg.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(out_dir, f"{batch_tag}_avg_only.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(avg_row, f, indent=2)

    return df

def summarize_dataset(dataset_root, output_root):
    batches = find_runs(dataset_root)
    if not batches:
        raise SystemExit(f"No runs found under: {dataset_root}")

    os.makedirs(output_root, exist_ok=True)
    per_batch_tables = []

    # Per-batch summaries
    for batch_tag, run_paths in batches.items():
        out_dir = os.path.join(output_root, batch_tag)
        df_batch = summarize_batch(batch_tag, run_paths, out_dir)
        per_batch_tables.append(df_batch[df_batch["run"] == "AVG_over_runs"])

    # Overall (avg across batch averages)
    overall_df = pd.concat(per_batch_tables, ignore_index=True)
    numeric_cols = [c for c in overall_df.columns if c not in ("batch", "run")]
    overall_avg = overall_df[numeric_cols].mean(numeric_only=True)
    overall_row = {"batch": "ALL_BATCHES", "run": "AVG_over_batches"}
    for k, v in overall_avg.items():
        overall_row[k] = float(v)

    overall_table = pd.concat([overall_df, pd.DataFrame([overall_row])], ignore_index=True)
    overall_csv = os.path.join(output_root, "OVERALL_batches_avg.csv")
    overall_table.to_csv(overall_csv, index=False)

    with open(os.path.join(output_root, "OVERALL_batches_avg.json"), "w", encoding="utf-8") as f:
        json.dump(overall_row, f, indent=2)

    print(f"[OK] Wrote per-batch summaries to {output_root}")
    print(f"[OK] Wrote overall summary to {overall_csv}")

# ---------- CLI ----------
def get_args():
    ap = argparse.ArgumentParser("Aggregate QK-PINN results (train/valid/test; per-batch and overall)")
    ap.add_argument("--dataset_root", type=str, required=True,
                    help="Root with batch-tag subfolders (e.g., results_qk/XJTU)")
    ap.add_argument("--output_root",  type=str, required=True,
                    help="Where to save aggregated CSV/JSON (e.g., results_qk_agg/XJTU)")
    return ap.parse_args()

if __name__ == "__main__":
    args = get_args()
    summarize_dataset(args.dataset_root, args.output_root)
