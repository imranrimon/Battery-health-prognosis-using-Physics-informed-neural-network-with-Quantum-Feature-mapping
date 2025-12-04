# qk/main_XJTU_qk.py
import os, sys, argparse, time, json, csv, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qk.model_qpin import PINN_QK
try:
    from dataloader.dataloader import XJTUdata
except ModuleNotFoundError:
    from dataloader import XJTUdata

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    try: torch.cuda.set_device(0)
    except Exception: pass
    _ = torch.empty(1, device='cuda'); torch.cuda.synchronize()
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


def load_xjtu_loaders(xjtu_batch: str, normalization: str, user_bs: int = 256):
    dl_args = argparse.Namespace(
        normalization_method=normalization,
        batch_size=user_bs, num_workers=0, shuffle=True, drop_last=False,
        pin_memory=False, seed=42, save_folder=None, log_dir=None
    )
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=dl_args)

    train_list, test_list = [], []
    for fname in os.listdir(root):
        if xjtu_batch in fname:
            if ('4' in fname) or ('8' in fname):
                test_list.append(os.path.join(root, fname))
            else:
                train_list.append(os.path.join(root, fname))
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader  = data.read_all(specific_path_list=test_list)
    return {'train': train_loader['train_2'],
            'valid': train_loader['valid_2'],
            'test' : test_loader['test_3']}


def train_epoch(model, loader, optimizer, alpha=0.7, beta=0.2, grad_clip=None):
    model.train()
    l_data_sum = l_pde_sum = l_mono_sum = 0.0
    n_batches = 0
    t0 = time.time()

    for x1, x2, y1, y2 in loader:
        x1, x2, y1, y2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True), \
                         y1.to(DEVICE, non_blocking=True), y2.to(DEVICE, non_blocking=True)

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

        l_data_sum += l_data.item()
        l_pde_sum  += l_pde.item()
        l_mono_sum += l_mono.item()
        n_batches  += 1

    dt = time.time() - t0
    tr_log = {
        'L_data': l_data_sum / max(1, n_batches),
        'L_PDE' : l_pde_sum  / max(1, n_batches),
        'L_mono': l_mono_sum / max(1, n_batches),
    }
    tr_log['loss'] = tr_log['L_data'] + alpha*tr_log['L_PDE'] + beta*tr_log['L_mono']
    tr_log['secs'] = dt
    return tr_log


def eval_loader(model, loader, alpha=0.7, beta=0.2):
    model.eval()
    total = 0.0
    n_batches = 0
    for x1, x2, y1, y2 in loader:
        x1, x2, y1, y2 = x1.to(DEVICE, non_blocking=True), x2.to(DEVICE, non_blocking=True), \
                         y1.to(DEVICE, non_blocking=True), y2.to(DEVICE, non_blocking=True)
        with torch.enable_grad():
            u1, u1_t, _, g1 = model(x1)
            u2, u2_t, _, g2 = model(x2)
            l_data = 0.5*F.mse_loss(u1, y1) + 0.5*F.mse_loss(u2, y2)
            l_pde  = 0.5*F.mse_loss(u1_t, g1) + 0.5*F.mse_loss(u2_t, g2)
            l_mono = torch.relu((u2 - u1) * (y1 - y2)).sum()
            loss   = l_data + alpha*l_pde + beta*l_mono
        total += loss.item(); n_batches += 1
    return total / max(1, n_batches)


@torch.no_grad()
def _to_numpy(*tensors):
    out = []
    for t in tensors:
        if t is None: out.append(None); continue
        out.append(t.detach().cpu().numpy())
    return out


def collect_and_save_preds(model, loader, split: str, save_dir: str, alpha=0.7, beta=0.2):
    """
    Saves:
      - {split}_preds.npz with arrays: u_pred, y_true, u_t_pred, g_pred, residual
      - {split}_preds.csv (u_pred,y_true,residual) for quick look
    """
    model.eval()
    u_list, y_list = [], []
    ut_list, g_list, res_list = [], [], []

    for x1, x2, y1, y2 in loader:
        x1, x2, y1, y2 = x1.to(DEVICE), x2.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE)

        # sample 1
        with torch.enable_grad():
            u1, u1_t, _, g1 = model(x1)
        r1 = u1_t - g1

        # sample 2
        with torch.enable_grad():
            u2, u2_t, _, g2 = model(x2)
        r2 = u2_t - g2

        u = torch.cat([u1, u2], dim=0)
        y = torch.cat([y1, y2], dim=0)
        ut = torch.cat([u1_t, u2_t], dim=0)
        g  = torch.cat([g1, g2], dim=0)
        r  = torch.cat([r1, r2], dim=0)

        u_np, y_np, ut_np, g_np, r_np = _to_numpy(u, y, ut, g, r)
        u_list.append(u_np); y_list.append(y_np)
        ut_list.append(ut_np); g_list.append(g_np); res_list.append(r_np)

    u_all  = np.vstack(u_list) if u_list else np.empty((0,1))
    y_all  = np.vstack(y_list) if y_list else np.empty((0,1))
    ut_all = np.vstack(ut_list) if ut_list else np.empty((0,1))
    g_all  = np.vstack(g_list) if g_list else np.empty((0,1))
    r_all  = np.vstack(res_list) if res_list else np.empty((0,1))

    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, f"{split}_preds.npz")
    np.savez_compressed(npz_path,
                        u_pred=u_all, y_true=y_all,
                        u_t_pred=ut_all, g_pred=g_all, residual=r_all)

    # quick CSV with three columns (flat)
    csv_path = os.path.join(save_dir, f"{split}_preds.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['u_pred', 'y_true', 'residual'])
        for i in range(u_all.shape[0]):
            w.writerow([float(u_all[i,0]), float(y_all[i,0]), float(r_all[i,0])])

    return {'npz': npz_path, 'csv': csv_path}


def get_args():
    ap = argparse.ArgumentParser("Quantum-kernel PINN on XJTU (logging + preds)")
    ap.add_argument('--xjtu_batch', type=str, default='2C',
                    choices=['2C','3C','R2.5','R3','RW','satellite'])
    ap.add_argument('--normalization', type=str, default='min-max')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=0.7)
    ap.add_argument('--beta',  type=float, default=0.2)
    ap.add_argument('--M', type=int, default=256)
    ap.add_argument('--qk_qubits', type=int, default=8)
    ap.add_argument('--qk_depth',  type=int, default=2)
    ap.add_argument('--save_dir', type=str, default='results_qk/XJTU')
    ap.add_argument('--landmarks', type=str, default='data/qk_landmarks_xjtu.pt')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grad_clip', type=float, default=None)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--lr_patience', type=int, default=10)
    ap.add_argument('--lr_factor', type=float, default=0.5)
    return ap.parse_args()


def set_seed(s: int):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.landmarks), exist_ok=True)
    set_seed(args.seed)

    loaders = load_xjtu_loaders(args.xjtu_batch, args.normalization, user_bs=args.batch_size)

    # landmarks from first train batch (drop time col)
    x1, x2, _, _ = next(iter(loaders['train']))
    lm = torch.cat([x1[:args.M, :-1], x2[:args.M, :-1]], dim=0)[:args.M].contiguous()


    model = PINN_QK(landmarks=lm, M=lm.size(0),
                    alpha=args.alpha, beta=args.beta,
                    qk_qubits=args.qk_qubits, qk_depth=args.qk_depth,
                    device=DEVICE).to(DEVICE)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    try:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True
        )
    except TypeError:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=args.lr_factor, patience=args.lr_patience
        )

    # --- history & best trackers ---
    hist_csv = os.path.join(args.save_dir, 'history.csv')
    if not os.path.exists(hist_csv):
        with open(hist_csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['epoch','train_loss','train_L_data','train_L_PDE','train_L_mono','val_loss','secs','lr'])

    best = float('inf'); best_ep = 0

    for ep in range(1, args.epochs + 1):
        tr = train_epoch(model, loaders['train'], optim,
                         alpha=args.alpha, beta=args.beta, grad_clip=args.grad_clip)
        val = eval_loader(model, loaders['valid'], alpha=args.alpha, beta=args.beta)
        sched.step(val)
        lr_now = optim.param_groups[0]['lr']

        print(f"Epoch {ep:03d} | train {tr['loss']:.6f} "
              f"(data {tr['L_data']:.6f}, pde {tr['L_PDE']:.6f}, mono {tr['L_mono']:.6f}, {tr['secs']:.1f}s) "
              f"| val {val:.6f} | lr {lr_now:.2e}")

        # append row to history.csv
        with open(hist_csv, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([ep, tr['loss'], tr['L_data'], tr['L_PDE'], tr['L_mono'], val, tr['secs'], lr_now])

        # save "last"
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'last_qk.pt'))

        # track & save "best"
        if val < best:
            best = val; best_ep = ep
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_qk.pt'))
            with open(os.path.join(args.save_dir, 'best_epoch.json'), 'w') as f:
                json.dump({
                    'epoch': best_ep,
                    'val_loss': best,
                    'train': tr,
                    'lr': lr_now
                }, f, indent=2)

        # early stop
        if ep - best_ep >= args.patience:
            print(f"Early stopping at epoch {ep} (no val improvement for {args.patience} epochs).")
            break

    print(f"Done. Best val: {best:.6f} at epoch {best_ep}")

    # --------- save predictions for train/valid/test ----------
    print("[DBG] saving predictions…")
    for split in ['train','valid','test']:
        paths = collect_and_save_preds(model, loaders[split], split, args.save_dir,
                                       alpha=args.alpha, beta=args.beta)
        print(f"[DBG] {split} -> {paths['npz']} | {paths['csv']}")

if __name__ == '__main__':
    main()