    try: torch.cuda.set_device(0)
    except Exception: pass
    _ = torch.empty(1, device='cuda'); torch.cuda.synchronize()

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# -------------- utils --------------
def makedirs(p): os.makedirs(p, exist_ok=True)

def now_tag(): return time.strftime("%Y%m%d_%H%M%S")

def write_json(obj, path):
    makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def append_csv_row(path, header, row_dict):
    makedirs(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row_dict)

def save_pred_csv(path, cols, rows):
    makedirs(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def write_time_row(csv_path, header, row):
    makedirs(os.path.dirname(csv_path))
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row)

# -------------- metrics helpers (optional global summary) --------------
def _safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def _r2(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot

def compute_pointwise_metrics(pairs):
    # pairs: list of (t, y_true, y_pred)
    if not pairs:
        return {"MSE": None, "RMSE": None, "MAPE": None, "R2": None, "N": 0}
    y = np.array([r[1] for r in pairs], dtype=np.float64)
    p = np.array([r[2] for r in pairs], dtype=np.float64)
    mse = float(np.mean((y - p) ** 2))
    rmse = float(math.sqrt(mse))
    mape = float(_safe_mape(y, p))
    r2 = float(_r2(y, p))
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2, "N": int(len(y))}

# -------------- data ---------------
def load_xjtu_loaders(xjtu_batch: str, normalization: str, user_bs: int = 256,
                      save_folder: str = None, log_dir: str = None):
    """
    Returns { 'train': train_2, 'valid': valid_2, 'test': test_3 } yielding (x1,x2,y1,y2).
    Split rule from original code:
      - choose files whose name contains the batch tag (e.g., '2C', '3C', 'R2.5', ...)
      - files with '4' or '8' in the filename -> test; others -> train
    """
    dl_args = argparse.Namespace(
        normalization_method=normalization,
        batch_size=user_bs,
        num_workers=0, shuffle=True, drop_last=False, pin_memory=False, seed=42,
        save_folder=save_folder, log_dir=log_dir
    )
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=dl_args)

    train_list, test_list = [], []
    for fname in os.listdir(root):
        if not fname.lower().endswith('.csv'):  # XJTU might be .csv; keep flexible if needed
            continue
        if xjtu_batch in fname:
            fp = os.path.join(root, fname)
            if ('4' in fname) or ('8' in fname):
                test_list.append(fp)
            else:
                train_list.append(fp)

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader  = data.read_all(specific_path_list=test_list)
    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test' : test_loader['test_3'],
    }

# --------- train / eval ----------
def train_epoch(model, loader, optimizer, alpha=0.7, beta=0.2, grad_clip=None):
    """
    loss = L_data + α * L_PDE + β * L_mono
    batches yield (x1, x2, y1, y2)
    """
    model.train()
    l1 = l2 = l3 = 0.0; n = 0
    _cuda_sync(); t0 = time.time()

    for x1, x2, y1, y2 in loader:
        x1, x2, y1, y2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True), \
                         y1.to(DEVICE, non_blocking=True), y2.to(DEVICE, non_blocking=True)

        # forward: (u, u_t, u_x, g). For XJTU we don't use u_x in loss.
        u1, u1_t, _, g1 = model(x1)
        u2, u2_t, _, g2 = model(x2)

        l_data = 0.5*F.mse_loss(u1, y1) + 0.5*F.mse_loss(u2, y2)
        l_pde  = 0.5*F.mse_loss(u1_t, g1) + 0.5*F.mse_loss(u2_t, g2)
        l_mono = torch.relu((u2 - u1) * (y1 - y2)).sum()

        loss = l_data + alpha*l_pde + beta*l_mono
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        l1 += l_data.item(); l2 += l_pde.item(); l3 += l_mono.item(); n += 1

    _cuda_sync(); secs = time.time() - t0
    return {
        'L_data': l1/max(1,n), 'L_PDE': l2/max(1,n), 'L_mono': l3/max(1,n),
        'loss': (l1 + alpha*l2 + beta*l3)/max(1,n), 'secs': secs
    }

@torch.no_grad()
def eval_loader(model, loader, alpha=0.7, beta=0.2, return_preds=False):
    model.eval()
    total = 0.0; n = 0; preds = []
    for x1, x2, y1, y2 in loader:
        x1, x2, y1, y2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True), \
                         y1.to(DEVICE, non_blocking=True), y2.to(DEVICE, non_blocking=True)

        # need grads for u_t inside model? model computes u_t internally; no backward here
        u1, u1_t, _, g1 = model(x1)
        u2, u2_t, _, g2 = model(x2)

        l_data = 0.5*F.mse_loss(u1, y1) + 0.5*F.mse_loss(u2, y2)
        l_pde  = 0.5*F.mse_loss(u1_t, g1) + 0.5*F.mse_loss(u2_t, g2)
        l_mono = torch.relu((u2 - u1) * (y1 - y2)).sum()
        loss   = l_data + alpha*l_pde + beta*l_mono

        total += loss.item(); n += 1

        if return_preds:
            t1 = x1[:, -1:].detach().cpu().numpy().ravel()
            t2 = x2[:, -1:].detach().cpu().numpy().ravel()
            y1n = y1.detach().cpu().numpy().ravel()
            y2n = y2.detach().cpu().numpy().ravel()
            u1n = u1.detach().cpu().numpy().ravel()
            u2n = u2.detach().cpu().numpy().ravel()
            preds += list(zip(t1, y1n, u1n)); preds += list(zip(t2, y2n, u2n))

    avg = total/max(1,n)
    return (avg, preds) if return_preds else avg

# ------------- CLI / driver -------------
def get_args():
    ap = argparse.ArgumentParser("Quantum-kernel PINN on XJTU (timed, logged)")
    ap.add_argument('--xjtu_batch', type=str, default='2C',
                    choices=['2C','3C','R2.5','R3','RW','satellite'])
    ap.add_argument('--normalization', type=str, default='min-max')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--beta',  type=float, default=0.2)
    # scheduler
    ap.add_argument('--lr_factor',   type=float, default=0.5)
    ap.add_argument('--lr_patience', type=int,   default=10)
    ap.add_argument('--patience',    type=int,   default=30)   # early stop
    # quantum kernel / landmarks
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--qk_qubits', type=int, default=8)
    ap.add_argument('--qk_depth',  type=int, default=2)
    # io / misc
    ap.add_argument('--save_root', type=str, default='results_qk/XJTU')
    ap.add_argument('--run_tag', type=str, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grad_clip', type=float, default=None)
    return ap.parse_args()

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def main():
    args = get_args()
    set_seed(args.seed)

    # per-batch save dir
    sub = f"{args.xjtu_batch}"
    run = args.run_tag or f"{now_tag()}_seed{args.seed}"
    save_dir = os.path.join(args.save_root, sub, run)
    makedirs(save_dir)
    print(f"[RUN] save_dir = {save_dir}")
    write_json(vars(args), os.path.join(save_dir, "config.json"))

    # Data
    print("[DBG] building XJTU loaders…", flush=True)
    loaders = load_xjtu_loaders(args.xjtu_batch, args.normalization, user_bs=args.batch_size)
    print("[DBG] loaders ready.", flush=True)

    # Landmarks from first train batch (drop last col t)
    x1, x2, _, _ = next(iter(loaders['train']))
    lm = torch.cat([x1[:args.M, :-1], x2[:args.M, :-1]], dim=0)[:args.M].contiguous()
    print(f"[DBG] landmarks: M={lm.size(0)} d={lm.size(1)}")

    # Model
    print("[DBG] constructing model…", flush=True)
    model = PINN_QK(landmarks=lm, M=lm.size(0),
                    alpha=args.alpha, beta=args.beta,
                    qk_qubits=args.qk_qubits, qk_depth=args.qk_depth,
                    device=DEVICE).to(DEVICE)
    print("[DBG] model ready.", flush=True)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    try:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=args.lr_factor, patience=args.lr_patience
        )
    except TypeError:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min')

    # per-epoch metrics CSV
    metrics_csv = os.path.join(save_dir, "metrics.csv")
    header = ["epoch","train_total","train_L_data","train_L_PDE","train_L_mono","val_total","secs","lr"]

    # ---- training loop with total timing ----
    best = float('inf'); best_ep = 0
    _cuda_sync(); t_run0 = time.time()

    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, loaders['train'], optim,
                         alpha=args.alpha, beta=args.beta, grad_clip=args.grad_clip)
        val = eval_loader(model, loaders['valid'], alpha=args.alpha, beta=args.beta)
        sched.step(val)

        lr_now = optim.param_groups[0]['lr']
        print(f"Epoch {ep:03d} | train {tr['loss']:.6f} "
              f"(data {tr['L_data']:.6f}, pde {tr['L_PDE']:.6f}, mono {tr['L_mono']:.6f}, {tr['secs']:.1f}s) "
              f"| val {val:.6f} | lr {lr_now:.2e}")

        # save rolling artifacts
        torch.save(model.state_dict(), os.path.join(save_dir, "last_qk.pt"))
        append_csv_row(metrics_csv, header, {
            "epoch": ep, "train_total": tr['loss'], "train_L_data": tr['L_data'],
            "train_L_PDE": tr['L_PDE'], "train_L_mono": tr['L_mono'],
            "val_total": val, "secs": tr['secs'], "lr": lr_now
        })

        if val < best:
            best = val; best_ep = ep
            torch.save(model.state_dict(), os.path.join(save_dir, "best_qk.pt"))
            write_json({"best_val": best, "best_epoch": best_ep},
                       os.path.join(save_dir, "best_metrics.json"))

        if ep - best_ep >= args.patience:
            print(f"Early stopping at epoch {ep} (no val improvement for {args.patience} epochs).")
            break

    _cuda_sync(); total_wall_s = time.time() - t_run0
    epochs_run = ep
    avg_epoch_s = total_wall_s / max(1, epochs_run)
    print(f"[TIME] total_wall_s={total_wall_s:.2f}  avg_epoch_s={avg_epoch_s:.2f}")

    # --- write timing summaries ---
    time_row = {
        "batch_tag": args.xjtu_batch,
        "save_dir": save_dir,
        "epochs_run": epochs_run,
        "total_wall_s": float(total_wall_s),
        "avg_epoch_s": float(avg_epoch_s),
    }
    # per-run summary inside run folder
    write_time_row(os.path.join(save_dir, "time_summary.csv"),
                   ["batch_tag","save_dir","epochs_run","total_wall_s","avg_epoch_s"], time_row)
    # aggregate to per-batch folder
    write_time_row(os.path.join(args.save_root, args.xjtu_batch, "TIMES.csv"),
                   ["batch_tag","save_dir","epochs_run","total_wall_s","avg_epoch_s"], time_row)

    # ----- predictions (valid & test) with best weights -----
    best_path = os.path.join(save_dir, "best_qk.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.to(DEVICE)

    vloss, vpreds = eval_loader(model, loaders['valid'], alpha=args.alpha, beta=args.beta, return_preds=True)
    tloss, tpreds = eval_loader(model, loaders['test'],  alpha=args.alpha, beta=args.beta, return_preds=True)

    save_pred_csv(os.path.join(save_dir, "preds_valid.csv"),
                  cols=["t","y_true","y_pred"],
                  rows=[[float(t), float(y), float(p)] for (t,y,p) in vpreds])
    save_pred_csv(os.path.join(save_dir, "preds_test.csv"),
                  cols=["t","y_true","y_pred"],
                  rows=[[float(t), float(y), float(p)] for (t,y,p) in tpreds])

    # optional: global metrics from predictions
    vm = compute_pointwise_metrics(vpreds)
    tm = compute_pointwise_metrics(tpreds)

    write_json({
        "best_epoch": best_ep, "best_val": best,
        "final_val": vloss, "final_test": tloss,
        "valid_metrics": vm, "test_metrics": tm,
        "save_dir": save_dir
    }, os.path.join(save_dir, "summary.json"))

    print(f"[OK] Wrote logs, timings & predictions to {save_dir}")

if __name__ == '__main__':
    main()
