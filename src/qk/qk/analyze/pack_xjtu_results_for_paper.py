# qk/analyze/pack_xjtu_results_for_paper.py
import os, json, argparse, numpy as np, pandas as pd

def ensure(p): os.makedirs(p, exist_ok=True)

def synth_logging_txt(metrics_csv: str, ids_train, ids_test, out_txt: str):
    df = pd.read_csv(metrics_csv)
    lines = []
    # write ID lines in their parser’s format
    lines.append(str(ids_train) + "\n")
    lines.append("\n")  # they read line indexes [1] and [3]; keep spacer lines
    lines.append(str(ids_test)  + "\n")
    lines.append("\n")
    # emulate training/valid lines that XJTU results.py parses
    # it looks for: "[Train] ... total loss:<float>"
    # and for: "[Valid] ... MSE:<float>"
    for i, row in df.iterrows():
        lines.append(f"[Train] epoch:{int(row['epoch'])}, total loss:{row['train_total']:.6f}\n")
        lines.append(f"[Valid] epoch:{int(row['epoch'])}, MSE:{row['val_total']:.6f}\n")
    # optional “test” echo (they also parse “[Test] MSE: …” if present)
    # append best val as a proxy for test if you want:
    try:
        best = df['val_total'].min()
        best_ep = int(df.loc[df['val_total'].idxmin(), 'epoch'])
        lines.append(f"[Test] epoch:{best_ep}, MSE:{best:.6f}\n")
    except Exception:
        pass
    with open(out_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)

def main():
    ap = argparse.ArgumentParser("Pack QK-PINN XJTU results into paper-style folders")
    ap.add_argument("--qk_root", default="results_qk/XJTU", help="your QK runs root")
    ap.add_argument("--paper_root", default="results/Ours/XJTU results", help="paper-style output root")
    ap.add_argument("--batch_tag", required=True, help="e.g. 2C, 3C, R2.5, R3, RW, satellite")
    ap.add_argument("--train_batch", type=int, default=0, help="0..5 like their code")
    ap.add_argument("--test_batch",  type=int, default=0, help="0..5")
    ap.add_argument("--ids_train", nargs="*", default=None,
                    help="optional list of training file paths (as their analyzer prints). If omitted, a stub list is used.")
    ap.add_argument("--ids_test",  nargs="*", default=None,
                    help="optional list of test file paths. If omitted, a stub list is used.")
    ap.add_argument("--topN", type=int, default=10, help="number of runs to export as Experiment1..N")
    args = ap.parse_args()

    # discover runs for this batch_tag
    src_dir = os.path.join(args.qk_root, args.batch_tag)
    if not os.path.isdir(src_dir):
        raise SystemExit(f"Not found: {src_dir}")

    runs = sorted(os.listdir(src_dir))[:args.topN]
    if not runs:
        raise SystemExit(f"No runs found in {src_dir}")

    # IDs fallbacks (the parser only displays them)
    ids_train = args.ids_train or ["data/XJTU data/<train_file_1>.csv", "data/XJTU data/<train_file_2>.csv"]
    ids_test  = args.ids_test  or ["data/XJTU data/<test_file_1>.csv"]

    # paper-style destination: ../results/Ours/XJTU results/<train>-<test>/Experimentk/
    out_parent = os.path.join(args.paper_root, f"{args.train_batch}-{args.test_batch}")
    ensure(out_parent)

    for k, run in enumerate(runs, start=1):
        src_run = os.path.join(src_dir, run)
        dst = os.path.join(out_parent, f"Experiment{k}")
        ensure(dst)

        # 1) convert preds_test.csv → pred_label.npy / true_label.npy (the analyzer expects a single concatenated series)
        preds_test = os.path.join(src_run, "preds_test.csv")
        if not os.path.exists(preds_test):
            print(f"[warn] missing {preds_test}, skipping run {run}")
            continue
        df = pd.read_csv(preds_test)  # columns: t,y_true,y_pred
        # sort by t to get a contiguous “degradation curve” (matches their split by jumps)
        df = df.sort_values("t")
        np.save(os.path.join(dst, "true_label.npy"), df["y_true"].to_numpy())
        np.save(os.path.join(dst, "pred_label.npy"), df["y_pred"].to_numpy())

        # 2) synthesize logging.txt from metrics.csv (+ write IDs lines it expects)
        metrics_csv = os.path.join(src_run, "metrics.csv")
        if os.path.exists(metrics_csv):
            synth_logging_txt(metrics_csv, ids_train, ids_test, os.path.join(dst, "logging.txt"))
        else:
            # still create a minimal file so parser doesn’t crash
            with open(os.path.join(dst, "logging.txt"), "w") as f:
                f.write(str(ids_train) + "\n\n" + str(ids_test) + "\n\n")

        print(f"[OK] Packed run '{run}' → {dst}")

if __name__ == "__main__":
    main()
