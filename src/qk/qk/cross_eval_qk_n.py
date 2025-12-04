# qk/cross_eval_qk.py
import os, sys, time, json, csv, argparse, copy, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from qk.model_qpin import PINN_QK

try:
    from dataloader.dataloader import XJTUdata, TJUdata, MITdata, HUSTdata
except ModuleNotFoundError:
    from dataloader import XJTUdata, TJUdata, MITdata, HUSTdata

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    try: torch.cuda.set_device(0)
    except Exception: pass
    _ = torch.empty(1, device='cuda'); torch.cuda.synchronize()

# ── utils ─────────────────────────────────────────────────────────────────────
def makedirs(p): os.makedirs(p, exist_ok=True)
def write_json(obj, path):
    makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
def append_row(path, header, row):
    makedirs(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row)

def _safe_mape(y, p, eps=1e-8):
    y = np.asarray(y); p = np.asarray(p)
    d = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs((y - p) / d)) * 100.0)
def _r2(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ss_res = np.sum((y - p)**2)
    ss_tot = np.sum((y - np.mean(y))**2) + 1e-12
    return float(1.0 - ss_res/ss_tot)
def compute_metrics(y, p):
    y = np.asarray(y).ravel()
    p = np.asarray(p).ravel()
    mse  = float(np.mean((y - p)**2))
    rmse = float(np.sqrt(mse))
    mape = _safe_mape(y, p)
    r2   = _r2(y, p)
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
def _gpu_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0.0

# ── loaders ───────────────────────────────────────────────────────────────────
def _dl_args(norm, bs):
    return argparse.Namespace(
        normalization_method=norm, batch_size=bs, num_workers=0,
        shuffle=True, drop_last=False, pin_memory=False, seed=42,
        save_folder=None, log_dir=None
    )

def build_loaders(dataset, norm="min-max", bs=512,
                  xjtu_batch="2C", tju_batch=0):
    if dataset == "XJTU":
        root = 'data/XJTU data'
        data = XJTUdata(root=root, args=_dl_args(norm, bs))
        train_list, test_list = [], []
        for f in os.listdir(root):
            if xjtu_batch in f:
                if ('4' in f) or ('8' in f):
                    test_list.append(os.path.join(root, f))
                else:
                    train_list.append(os.path.join(root, f))
        tr = data.read_all(specific_path_list=train_list)
        te = data.read_all(specific_path_list=test_list)
        return {'train': tr['train_2'], 'valid': tr['valid_2'], 'test': te['test_3']}

    if dataset == "TJU":
        root = 'data/TJU data'
        data = TJUdata(root=root, args=_dl_args(norm, bs))
        batches = sorted(os.listdir(root))
        bname = batches[tju_batch]
        files = sorted(os.path.join(root, bname, f) for f in os.listdir(os.path.join(root, bname)))
        mod = [(5, 9), (4, 8), (5, 9)][tju_batch]
        train_list, test_list = [], []
        for i, fp in enumerate(files):
            idx1 = i + 1
            if idx1 % 10 in mod: test_list.append(fp)
            else:                train_list.append(fp)
        tr = data.read_all(specific_path_list=train_list)
        te = data.read_all(specific_path_list=test_list)
        return {'train': tr['train_2'], 'valid': tr['valid_2'], 'test': te['test_3']}

    if dataset == "MIT":
        root = 'data/MIT data'
        data = MITdata(root=root, args=_dl_args(norm, bs))
        ld = data.read_all(specific_path_list=None)
        return {'train': ld['train_2'], 'valid': ld['valid_2'], 'test': ld['test_3']}

    if dataset == "HUST":
        root = 'data/HUST data'
        data = HUSTdata(root=root, args=_dl_args(norm, bs))
        ld = data.read_all(specific_path_list=None)
        return {'train': ld['train_2'], 'valid': ld['valid_2'], 'test': ld['test_3']}

    raise ValueError(f"Unknown dataset {dataset}")

# ── subset controls ───────────────────────────────────────────────────────────
def _subset_loader(loader, max_items: int, seed: int):
    if max_items is None or max_items <= 0:
        return loader
    ds = loader.dataset
    n = len(ds)
    k = min(max_items, n)
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    idx = idx[:k]
    subset = Subset(ds, idx)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=getattr(loader, "num_workers", 0),
        pin_memory=getattr(loader, "pin_memory", False),
        drop_last=getattr(loader, "drop_last", False),
    )

def shrink_loaders(loaders: dict, lt: int, lv: int, le: int, seed: int):
    return {
        'train': _subset_loader(loaders['train'], lt, seed),
        'valid': _subset_loader(loaders['valid'], lv, seed+1),
        'test' : _subset_loader(loaders['test'],  le, seed+2),
    }

# ── train / eval ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optim, alpha, beta, grad_clip=None, use_amp=False):
    model.train()
    d = p = m = 0.0; n = 0
    _cuda_sync()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for x1, x2, y1, y2 in loader:
        x1, x2, y1, y2 = x1.to(DEVICE), x2.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=use_amp):
            u1, u1_t, _, g1 = model(x1)
            u2, u2_t, _, g2 = model(x2)
            l_data = 0.5*F.mse_loss(u1, y1) + 0.5*F.mse_loss(u2, y2)
            l_pde  = 0.5*F.mse_loss(u1_t, g1) + 0.5*F.mse_loss(u2_t, g2)
            l_mono = torch.relu((u2 - u1) * (y1 - y2)).sum()
            loss = l_data + alpha*l_pde + beta*l_mono

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optim)
        scaler.update()

        d += l_data.item(); p += l_pde.item(); m += l_mono.item(); n += 1

    _cuda_sync()
    secs = time.time() - t0
    peak_mb = _gpu_peak_mb()
    Ld, Lp, Lm = d/max(1,n), p/max(1,n), m/max(1,n)
    return {'loss': Ld + alpha*Lp + beta*Lm, 'L_data': Ld, 'L_PDE': Lp, 'L_mono': Lm,
            'secs': secs, 'gpu_mb': peak_mb}

@torch.no_grad()
def eval_split_metrics(model, loader, use_amp=False):
    model.eval()
    y_all, u_all = [], []
    for x1, x2, y1, y2 in loader:
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=use_amp):
            u1, _, _, _ = model(x1)
            u2, _, _, _ = model(x2)
        y_all.append(torch.cat([y1, y2], 0).cpu().numpy())
        u_all.append(torch.cat([u1, u2], 0).detach().cpu().numpy())
    if not y_all:
        return dict(MSE=np.nan, RMSE=np.nan, MAPE=np.nan, R2=np.nan)
    y = np.vstack(y_all); u = np.vstack(u_all)
    return compute_metrics(y, u)

def fit_qpinn(loaders, args, save_dir, landmarks_M=256):
    """Train on source loaders; return (model, landmarks) for later re-instantiation."""
    x1, x2, _, _ = next(iter(loaders['train']))
    landmarks = torch.cat([x1[:landmarks_M, :-1], x2[:landmarks_M, :-1]], dim=0)[:landmarks_M].contiguous()

    model = PINN_QK(landmarks=landmarks, M=landmarks.size(0),
                    alpha=args.alpha, beta=args.beta,
                    qk_qubits=args.qk_qubits, qk_depth=args.qk_depth,
                    device=DEVICE).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    hist_csv = os.path.join(save_dir, "train_history.csv")
    makedirs(save_dir)
    if not os.path.exists(hist_csv):
        with open(hist_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train","L_data","L_PDE","L_mono",
                                    "secs_epoch","gpu_peak_mb","lr"])

    for ep in range(1, args.epochs+1):
        tr = train_epoch(model, loaders['train'], optim, args.alpha, args.beta,
                         grad_clip=args.grad_clip, use_amp=args.amp)
        with open(hist_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, tr['loss'], tr['L_data'], tr['L_PDE'], tr['L_mono'],
                                    tr['secs'], tr['gpu_mb'], optim.param_groups[0]['lr']])
        if ep % max(1, args.log_every) == 0 or ep == args.epochs:
            print(f"[SRC] epoch {ep:03d}  train={tr['loss']:.6f}  "
                  f"(data {tr['L_data']:.6f}, pde {tr['L_PDE']:.6f}, mono {tr['L_mono']:.6f})  "
                  f"{tr['secs']:.2f}s  GPU~{tr['gpu_mb']:.0f}MB")
    torch.save(model.state_dict(), os.path.join(save_dir, "source_final_qk.pt"))
    return model, landmarks

def _set_trainable_modules(model: torch.nn.Module, names_to_train):
    names_to_train = set(names_to_train or [])
    for name, p in model.named_parameters():
        root = name.split('.')[0]
        p.requires_grad = (root in names_to_train)
    if not any(p.requires_grad for p in model.parameters()):
        for p in model.parameters():
            p.requires_grad = True

def finetune_on_target(source_state_dict, landmarks, loaders_tgt, args, save_dir):
    """Create a fresh model with same landmarks, load source weights, freeze modules, FT."""
    model_ft = PINN_QK(
        landmarks=landmarks, M=landmarks.size(0),
        alpha=args.alpha, beta=args.beta,
        qk_qubits=args.qk_qubits, qk_depth=args.qk_depth,
        device=DEVICE
    ).to(DEVICE)
    model_ft.load_state_dict(source_state_dict, strict=True)

    _set_trainable_modules(model_ft, args.ft_train_submodules)
    params = [p for p in model_ft.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.ft_lr)

    hist_csv = os.path.join(save_dir, "ft_history.csv")
    makedirs(save_dir)
    if not os.path.exists(hist_csv):
        with open(hist_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train","L_data","L_PDE","L_Mono",
                                    "secs_epoch","gpu_peak_mb","lr"])

    for ep in range(1, args.ft_epochs+1):
        tr = train_epoch(model_ft, loaders_tgt['train'], opt, args.alpha, args.beta,
                         grad_clip=args.grad_clip, use_amp=args.amp)
        with open(hist_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, tr['loss'], tr['L_data'], tr['L_PDE'], tr['L_mono'],
                                    tr['secs'], tr['gpu_mb'], opt.param_groups[0]['lr']])
        if ep % max(1, args.log_every) == 0 or ep == args.ft_epochs:
            print(f"[FT ] epoch {ep:03d}  train={tr['loss']:.6f}  "
                  f"{tr['secs']:.2f}s  GPU~{tr['gpu_mb']:.0f}MB")

    torch.save(model_ft.state_dict(), os.path.join(save_dir, "finetuned_qk.pt"))
    return model_ft

# ── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    ap = argparse.ArgumentParser("Cross-dataset evaluation for QPINN (source-only or fine-tune)")
    ap.add_argument("--source", choices=["XJTU","TJU","MIT","HUST"], required=True)
    ap.add_argument("--targets", nargs="*", default=["AUTO"],
                    help='List like XJTU TJU MIT HUST, or AUTO for "other three".')

    # training
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--normalization", type=str, default="min-max")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.7)
    ap.add_argument("--beta",  type=float, default=0.2)
    ap.add_argument("--grad_clip", type=float, default=None)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--qk_qubits", type=int, default=8)
    ap.add_argument("--qk_depth",  type=int, default=2)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--amp", action="store_true")

    # dataset knobs
    ap.add_argument("--xjtu_batch", type=str, default="2C")
    ap.add_argument("--tju_batch",  type=int, default=0, choices=[0,1,2])

    # subset limits
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_valid", type=int, default=None)
    ap.add_argument("--limit_test",  type=int, default=None)
    ap.add_argument("--subset_seed", type=int, default=123)

    # fine-tune
    ap.add_argument("--ft_epochs", type=int, default=0)
    ap.add_argument("--ft_lr",     type=float, default=5e-4)
    ap.add_argument("--ft_train_submodules", nargs="*", default=["enc_x", "solution_u"])

    # io
    ap.add_argument("--save_root", type=str, default="results_qk/cross_eval")
    ap.add_argument("--run_tag",   type=str, default=None)
    ap.add_argument("--seed",      type=int, default=42)
    return ap.parse_args()

def set_seed(s):
    import random as pyrand
    pyrand.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args(); set_seed(args.seed)
    all_ds = ["XJTU","TJU","MIT","HUST"]
    if args.targets == ["AUTO"]:
        targets = [d for d in all_ds if d != args.source]
    else:
        targets = [d for d in args.targets if d in all_ds and d != args.source]

    run_name = args.run_tag or f"{args.source}_to_{'-'.join(targets)}"
    out_dir = os.path.join(args.save_root, run_name)
    makedirs(out_dir)
    write_json(vars(args), os.path.join(out_dir, "config.json"))

    # source: build → shrink → train
    src_loaders = build_loaders(args.source, args.normalization, args.batch_size,
                                xjtu_batch=args.xjtu_batch, tju_batch=args.tju_batch)
    src_loaders = shrink_loaders(src_loaders,
                                 args.limit_train, args.limit_valid, args.limit_test,
                                 args.subset_seed)

    print(f"[TRAIN] source={args.source}; epochs={args.epochs}; out={out_dir}")
    model, landmarks = fit_qpinn(src_loaders, args, os.path.join(out_dir, f"source_{args.source}"), landmarks_M=args.M)

    # evaluate / fine-tune on each target
    summary_csv = os.path.join(out_dir, "summary.csv")
    header = ["source","target","mode","MSE","RMSE","MAPE","R2","save_dir"]
    src_state = model.state_dict()

    for tgt in targets:
        print(f"\n[→] Target={tgt}")
        tgt_loaders = build_loaders(tgt, args.normalization, args.batch_size,
                                    xjtu_batch=args.xjtu_batch, tju_batch=args.tju_batch)
        tgt_loaders = shrink_loaders(tgt_loaders,
                                     args.limit_train, args.limit_valid, args.limit_test,
                                     args.subset_seed)

        # source-only
        m_src_only = eval_split_metrics(model, tgt_loaders['test'], use_amp=args.amp)
        row_src = dict(source=args.source, target=tgt, mode="source_only",
                       save_dir=out_dir, **m_src_only)
        append_row(summary_csv, header, row_src)
        print(f"   source-only: RMSE={m_src_only['RMSE']:.4f}  R2={m_src_only['R2']:.4f}")

        # fine-tune (fresh model; no deepcopy)
        if args.ft_epochs > 0:
            ft_dir = os.path.join(out_dir, f"ft_{tgt}")
            makedirs(ft_dir)
            model_ft = finetune_on_target(src_state, landmarks, tgt_loaders, args, save_dir=ft_dir)
            m_ft = eval_split_metrics(model_ft, tgt_loaders['test'], use_amp=args.amp)
            row_ft = dict(source=args.source, target=tgt,
                          mode=f"finetune_{args.ft_epochs}ep({'-'.join(args.ft_train_submodules)})",
                          save_dir=ft_dir, **m_ft)
            append_row(summary_csv, header, row_ft)
            print(f"   fine-tuned:  RMSE={m_ft['RMSE']:.4f}  R2={m_ft['R2']:.4f}")

    print(f"\n[OK] Wrote summary: {summary_csv}")

if __name__ == "__main__":
    main()
