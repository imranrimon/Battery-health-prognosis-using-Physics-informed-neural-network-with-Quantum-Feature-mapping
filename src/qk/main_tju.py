    try:
        torch.cuda.set_device(0)
    except Exception:
        pass
    _ = torch.empty(1, device='cuda'); torch.cuda.synchronize()

try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass

# -------------- utils --------------
def makedirs(p): os.makedirs(p, exist_ok=True)

def now_tag(): return time.strftime("%Y%m%d_%H%M%S")

def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def append_csv_row(path, header, row_dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row_dict)

def save_pred_csv(path, cols, rows):
    makedirs(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(cols); w.writerows(rows)

# -------------- data ---------------
def load_TJU_loaders(args, small_sample=None):
    """
    Returns loaders dict {train, valid, test} each yielding (x1,x2,y1,y2).
    Mirrors the authors' TJU split logic.
    """
    root = 'data/TJU data'
    dl_args = argparse.Namespace(
        normalization_method=args.normalization,
        batch_size=args.batch_size,
        num_workers=0, shuffle=True, drop_last=False, pin_memory=False, seed=42,
        save_folder=None, log_dir=None
    )
    data = TJUdata(root=root, args=dl_args)

    # units-digit rules per batch (same as main_TJU.py)
    mod = [(5, 9), (4, 8), (5, 9)]

    if args.in_same_batch:
        batchs = sorted(os.listdir(root))
        batch_name = batchs[args.batch]
        batch_root = os.path.join(root, batch_name)
        files = sorted(os.listdir(batch_root))

        train_list, test_list = [], []
        for i, f in enumerate(files):
            idx1 = i + 1
            if idx1 % 10 in mod[args.batch]:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))

        if small_sample is not None:
            train_list = train_list[:small_sample]

        train_ld = data.read_all(specific_path_list=train_list)
        test_ld  = data.read_all(specific_path_list=test_list)
        return {'train': train_ld['train_2'], 'valid': train_ld['valid_2'], 'test': test_ld['test_3']}
    else:
        # cross-batch: train on args.train_batch, test on args.test_batch
        _ = sorted(os.listdir(root))  # not strictly needed; kept for parity
        train_ld = data.read_one_batch(args.train_batch)
        test_ld  = data.read_one_batch(args.test_batch)
        return {'train': train_ld['train_2'], 'valid': train_ld['valid_2'], 'test': test_ld['test_3']}

# --------- train / eval ----------
def _grad_penalty(u_x_1, u_x_2):
    # 0.5*(||u_x||^2 at sample 1 + ||u_x||^2 at sample 2), averaged over batch
    g1 = u_x_1.pow(2).sum(dim=1).mean()
    g2 = u_x_2.pow(2).sum(dim=1).mean()
    return 0.5 * (g1 + g2)

def train_epoch(model, loader, optimizer, alpha=1.0, beta=0.05, lambda_grad=0.0, grad_clip=None):
    model.train()
    l1 = l2 = l3 = lg = 0.0; n = 0; t0 = time.time()
    for x1, x2, y1, y2 in loader:
        x1, x2, y1, y2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True), \
                         y1.to(DEVICE, non_blocking=True), y2.to(DEVICE, non_blocking=True)

        # model should return: u, u_t, u_x, g
        u1, u1_t, u1_x, g1 = model(x1)
        u2, u2_t, u2_x, g2 = model(x2)

        l_data = 0.5*F.mse_loss(u1, y1) + 0.5*F.mse_loss(u2, y2)
        l_pde  = 0.5*F.mse_loss(u1_t, g1) + 0.5*F.mse_loss(u2_t, g2)
        l_mono = torch.relu((u2 - u1) * (y1 - y2)).sum()

        l_grad = _grad_penalty(u1_x, u2_x)  # NEW: smoothness penalty on u_x

        loss = l_data + alpha*l_pde + beta*l_mono + lambda_grad*l_grad
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        l1 += l_data.item(); l2 += l_pde.item(); l3 += l_mono.item(); lg += l_grad.item(); n += 1

    secs = time.time() - t0
    return {
        'L_data': l1/max(1,n), 'L_PDE': l2/max(1,n), 'L_mono': l3/max(1,n),
        'L_grad': lg/max(1,n),
        'loss': (l1 + alpha*l2 + beta*l3 + lambda_grad*lg)/max(1,n),
        'secs': secs
    }

def eval_loader(model, loader, alpha=1.0, beta=0.05, lambda_grad=0.0, return_preds=False):
    model.eval()
    tot = td = tp = tm = tg = 0.0; n = 0; preds = []
    for x1, x2, y1, y2 in loader:
        x1, x2, y1, y2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True), \
                         y1.to(DEVICE, non_blocking=True), y2.to(DEVICE, non_blocking=True)
        with torch.enable_grad():
            u1, u1_t, u1_x, g1 = model(x1)
            u2, u2_t, u2_x, g2 = model(x2)
            l_data = 0.5*F.mse_loss(u1, y1) + 0.5*F.mse_loss(u2, y2)
            l_pde  = 0.5*F.mse_loss(u1_t, g1) + 0.5*F.mse_loss(u2_t, g2)
            l_mono = torch.relu((u2 - u1) * (y1 - y2)).sum()
            l_grad = _grad_penalty(u1_x, u2_x)

            loss   = l_data + alpha*l_pde + beta*l_mono + lambda_grad*l_grad

        tot += loss.item(); td += l_data.item(); tp += l_pde.item(); tm += l_mono.item(); tg += l_grad.item()
        n += 1

        if return_preds:
            t1 = x1[:, -1:].detach().cpu().numpy().ravel()
            t2 = x2[:, -1:].detach().cpu().numpy().ravel()
            y1n = y1.detach().cpu().numpy().ravel()
            y2n = y2.detach().cpu().numpy().ravel()
            u1n = u1.detach().cpu().numpy().ravel()
            u2n = u2.detach().cpu().numpy().ravel()
            preds += list(zip(t1, y1n, u1n)); preds += list(zip(t2, y2n, u2n))

    avg = tot/max(1,n)
    logs = {
        'val_total': avg,
        'val_L_data': td/max(1,n),
        'val_L_PDE' : tp/max(1,n),
        'val_L_mono': tm/max(1,n),
        'val_L_grad': tg/max(1,n),
    }
    return (avg, preds, logs) if return_preds else (avg, logs)

# ------------- CLI / driver -------------
def get_args():
    ap = argparse.ArgumentParser("Quantum-kernel PINN on TJU (with u_x penalty)")
    # data/split
    ap.add_argument('--in_same_batch', action='store_true', default=True,
                    help='If set, split train/test within selected batch using units-digit rule')
    ap.add_argument('--batch', type=int, default=0, choices=[0,1,2], help='Batch index used when in_same_batch=True')
    ap.add_argument('--train_batch', type=int, default=0, choices=[0,1,2], help='Train batch when in_same_batch=False')
    ap.add_argument('--test_batch',  type=int, default=1, choices=[0,1,2], help='Test batch when in_same_batch=False')
    ap.add_argument('--normalization', type=str, default='min-max')
    ap.add_argument('--batch_size', type=int, default=512)

    # schedule
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr',     type=float, default=1e-3)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--lr_patience', type=int, default=10)
    ap.add_argument('--lr_factor',   type=float, default=0.5)

    # losses
    ap.add_argument('--alpha', type=float, default=1.0)   # PDE weight
    ap.add_argument('--beta',  type=float, default=0.05)  # monotone-in-time pair weight
    ap.add_argument('--lambda_grad', type=float, default=1e-3, help='weight on ||u_x||^2 penalty')

    # quantum kernel / landmarks
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--qk_qubits', type=int, default=6)
    ap.add_argument('--qk_depth',  type=int, default=1)

    # io
    ap.add_argument('--save_root', type=str, default='results_qk/TJU')
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

    # Build per-batch save dir
    sub = f"batch{args.batch}" if args.in_same_batch else f"{args.train_batch}-{args.test_batch}"
    run = args.run_tag or f"{now_tag()}_seed{args.seed}"
    save_dir = os.path.join(args.save_root, sub, run)
    makedirs(save_dir)
    print(f"[RUN] save_dir = {save_dir}")
    write_json(vars(args), os.path.join(save_dir, "config.json"))

    # Data
    print("[DBG] building TJU loaders…", flush=True)
    loaders = load_TJU_loaders(args)
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
            optim, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True
        )
    except TypeError:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=args.lr_factor, patience=args.lr_patience
        )

    # Metrics CSV (now includes L_grad)
    metrics_csv = os.path.join(save_dir, "metrics.csv")
    header = ["epoch","train_total","train_L_data","train_L_PDE","train_L_mono","train_L_grad",
              "val_total","val_L_data","val_L_PDE","val_L_mono","val_L_grad","secs","lr"]

    # Train
    best = float('inf'); best_ep = 0
    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, loaders['train'], optim,
                         alpha=args.alpha, beta=args.beta, lambda_grad=args.lambda_grad,
                         grad_clip=args.grad_clip)
        val, vlog = eval_loader(model, loaders['valid'],
                                alpha=args.alpha, beta=args.beta, lambda_grad=args.lambda_grad)
        sched.step(val)

        lr_now = optim.param_groups[0]['lr']
        print(f"Epoch {ep:03d} | train {tr['loss']:.6f} "
              f"(data {tr['L_data']:.6f}, pde {tr['L_PDE']:.6f}, mono {tr['L_mono']:.6f}, grad {tr['L_grad']:.6f}, {tr['secs']:.1f}s) "
              f"| val {val:.6f} | lr {lr_now:.2e}")

        torch.save(model.state_dict(), os.path.join(save_dir, "last_qk.pt"))
        append_csv_row(metrics_csv, header, {
            "epoch": ep, "train_total": tr['loss'], "train_L_data": tr['L_data'],
            "train_L_PDE": tr['L_PDE'], "train_L_mono": tr['L_mono'], "train_L_grad": tr['L_grad'],
            "val_total": val, "val_L_data": vlog['val_L_data'], "val_L_PDE": vlog['val_L_PDE'],
            "val_L_mono": vlog['val_L_mono'], "val_L_grad": vlog['val_L_grad'],
            "secs": tr['secs'], "lr": lr_now
        })

        if val < best:
            best = val; best_ep = ep
            torch.save(model.state_dict(), os.path.join(save_dir, "best_qk.pt"))
            write_json({"best_val": best, "best_epoch": best_ep}, os.path.join(save_dir, "best_metrics.json"))

        if ep - best_ep >= args.patience:
            print(f"Early stopping at epoch {ep} (no val improvement for {args.patience} epochs).")
            break

    print(f"Done. Best val {best:.6f} @ epoch {best_ep}")

    # ----- predictions dump (valid & test) with best weights -----
    best_path = os.path.join(save_dir, "best_qk.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.to(DEVICE)

    vloss, vpreds, vlog = eval_loader(model, loaders['valid'],
                                      alpha=args.alpha, beta=args.beta, lambda_grad=args.lambda_grad, return_preds=True)
    tloss, tpreds, tlog = eval_loader(model, loaders['test'],
                                      alpha=args.alpha, beta=args.beta, lambda_grad=args.lambda_grad, return_preds=True)

    save_pred_csv(os.path.join(save_dir, "preds_valid.csv"),
                  cols=["t","y_true","y_pred"],
                  rows=[[float(t), float(y), float(p)] for (t,y,p) in vpreds])
    save_pred_csv(os.path.join(save_dir, "preds_test.csv"),
                  cols=["t","y_true","y_pred"],
                  rows=[[float(t), float(y), float(p)] for (t,y,p) in tpreds])

    write_json({
        "best_epoch": best_ep, "best_val": best,
        "final_val": vloss, "final_test": tloss,
        "val_breakdown": vlog, "test_breakdown": tlog,
        "save_dir": save_dir
    }, os.path.join(save_dir, "summary.json"))

    print(f"[OK] Wrote logs & predictions to {save_dir}")

if __name__ == '__main__':
    main()
