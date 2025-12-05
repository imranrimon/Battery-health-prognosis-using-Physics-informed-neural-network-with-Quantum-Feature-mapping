# spikan_mains/spikan_main_XJTU.py
import importlib
import os, json, math, argparse, sys, time, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------------
# Force-load SPIKAN override as the 'Model' package (no repo edits)
# ---------------------------------------------------------------------
# PROJECT_ROOT is now 2 levels up from src/spikan/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERRIDE_INIT = PROJECT_ROOT / "spikan_override" / "Model" / "__init__.py"

if not OVERRIDE_INIT.exists():
    raise FileNotFoundError(f"SPIKAN override not found at {OVERRIDE_INIT}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location("Model", str(OVERRIDE_INIT))
model_pkg = importlib.util.module_from_spec(spec)
sys.modules["Model"] = model_pkg
assert spec.loader is not None
spec.loader.exec_module(model_pkg)

from src.dataloaders.dataloader import XJTUdata
from src.models.pinn import PINN

# ---------------------------------------------------------------------
# Data loading (from your original main_XJTU logic)
# ---------------------------------------------------------------------
def load_data(args, small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list, test_list = [], []
    for file in os.listdir(root):
        if args.batch in file:
            if ('4' in file) or ('8' in file):
                test_list.append(os.path.join(root, file))
            else:
                train_list.append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader  = data.read_all(specific_path_list=test_list)
    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test' : test_loader['test_3']
    }

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def eval_one_epoch(pinn, loader, device):
    pinn.eval()
    loss_mse = torch.nn.MSELoss()
    alpha = float(getattr(pinn, 'alpha', 1.0))
    total, n = 0.0, 0
    with torch.enable_grad():
        for (xt, _, y, _) in loader:
            xt, y = xt.to(device).float(), y.to(device).float()
            u_hat, f, _, _ = pinn.forward(xt)
            l_data = loss_mse(u_hat, y)
            l_pde  = (f**2).mean()
            loss = l_data + alpha * l_pde
            total += float(loss.item()) * xt.size(0)
            n += xt.size(0)
    return total / max(1, n)

def test_and_save(pinn, loader, save_folder):
    pinn.eval()
    preds, trues = [], []
    with torch.enable_grad():
        for (xt, _, y, _) in loader:
            xt = xt.to(pinn.device).float()
            u_hat, _, _, _ = pinn.forward(xt)
            preds.append(u_hat.detach().cpu().numpy())
            trues.append(y.numpy())
    preds = np.concatenate(preds, 0)
    trues = np.concatenate(trues, 0)

    ensure_dir(save_folder)
    np.save(os.path.join(save_folder, "pred_label.npy"), preds)
    np.save(os.path.join(save_folder, "true_label.npy"), trues)

    mae = float(np.mean(np.abs(preds - trues)))
    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    denom = np.sum((trues - np.mean(trues)) ** 2)
    r2 = float(1.0 - np.sum((preds - trues) ** 2) / denom) if denom > 1e-12 else 0.0
    mape = float(np.mean(np.abs(preds - trues) / np.clip(np.abs(trues), 1e-6, None)))
    with open(os.path.join(save_folder, "metrics.json"), "w") as f:
        json.dump({"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}, f, indent=2)
    return rmse, mape

def detect_target_bounds(loader, device, k_batches=3):
    """Peek a few batches to infer target scale."""
    mins, maxs = [], []
    with torch.no_grad():
        for i, (xt, _, y, _) in enumerate(loader):
            y = y.to(device).float()
            mins.append(float(y.min().cpu())); maxs.append(float(y.max().cpu()))
            if i+1 >= k_batches: break
    tmin, tmax = (min(mins), max(maxs)) if mins else (0.0, 1.0)
    if -0.05 <= tmin and tmax <= 1.05: return '0,1', tmin, tmax
    if -1.05 <= tmin <= -0.2 and 0.2 <= tmax <= 1.05: return '-1,1', tmin, tmax
    return 'none', tmin, tmax

# ---------------------------------------------------------------------
# Training loop with early stopping + best checkpoint
# ---------------------------------------------------------------------
def train_one_run(args, batch, exp_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dirs
    run_root = Path(args.save_root) / "XJTU" / batch / f"Experiment{exp_idx}"
    ckpt_dir = run_root / "checkpoints"
    ensure_dir(ckpt_dir)

    # args required by dataloader / model
    setattr(args, 'batch', batch)
    setattr(args, 'save_folder', str(run_root))
    if not hasattr(args, 'log_dir'):
        setattr(args, 'log_dir', None)

    # data
    loaders = load_data(args)

    # detect target bounds for activation inside the model
    bounds, tmin, tmax = detect_target_bounds(loaders['train'], device)
    setattr(args, 'target_bounds', bounds)
    setattr(args, 'target_min', tmin)
    setattr(args, 'target_max', tmax)
    print(f"[{batch} | Exp {exp_idx}] Detected target bounds: {bounds}  (min={tmin:.4f}, max={tmax:.4f})")

    # model
    pinn = PINN(args)
    pinn.train()

    # optimizer & sched are created inside PINN
    opt = pinn.optimizer
    scheduler = pinn.scheduler
    loss_mse = torch.nn.MSELoss()
    alpha = float(getattr(args, 'alpha', 0.7))
    beta  = float(getattr(args, 'beta', 20.0))
    beta_warm = int(getattr(args, 'beta_warmup_epochs', 30))  # NEW

    # early stopping config
    epochs = int(getattr(args, 'epochs', 200))
    patience = int(getattr(args, 'early_stop', 20))
    best_val = float('inf')
    best_path = ckpt_dir / "best.pt"
    hist = []

    no_improve = 0
    for ep in range(1, epochs + 1):
        pinn.train()
        ep_loss = 0.0
        seen = 0

        # ramp monotonic weight to avoid collapse early
        beta_eff = beta * min(1.0, ep / max(1, beta_warm))

        for (xt, _, y, bid) in loaders['train']:
            xt, y = xt.to(device).float(), y.to(device).float()
            u_hat, f, _, _ = pinn.forward(xt)

            l_data = loss_mse(u_hat, y)
            l_pde  = (f**2).mean()

            l_mono = torch.tensor(0.0, device=device)
            if beta_eff > 0 and bid is not None:
                idx = torch.argsort(xt[:, 16])  # time last
                diffs = u_hat[idx][1:] - u_hat[idx][:-1]
                l_mono = F.relu(diffs).mean()

            loss = l_data + alpha*l_pde + beta_eff*l_mono
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
            opt.step()

            ep_loss += float(loss.item()) * xt.size(0)
            seen += xt.size(0)

        scheduler.step()

        val_loss = eval_one_epoch(pinn, loaders['valid'], device)
        hist.append({"epoch": ep, "train_loss": ep_loss / max(1, seen), "val_loss": val_loss})

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            no_improve = 0
            torch.save({"epoch": ep, "model": pinn.state_dict(), "best_val": best_val, "args": vars(args)}, best_path)
        else:
            no_improve += 1

        print(f"[{batch} | Exp {exp_idx}] Epoch {ep:03d}  "
              f"train={ep_loss/max(1,seen):.4e}  valid={val_loss:.4e}  "
              f"best={best_val:.4e}  beta_eff={beta_eff:.2f}  patience={no_improve}/{patience}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {ep}.")
            break

    # save final model
    torch.save({"epoch": ep, "model": pinn.state_dict(), "best_val": best_val, "args": vars(args)},
               ckpt_dir / "final.pt")

    # load best for test
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        pinn.load_state_dict(state["model"])

    # test + save predictions + metrics
    rmse, mape = test_and_save(pinn, loaders['test'], str(run_root))

    # write logs
    with open(run_root / "train_log.jsonl", "w") as f:
        for row in hist: f.write(json.dumps(row) + "\n")
    with open(run_root / "run_info.json", "w") as f:
        json.dump({"batch": batch, "experiment": exp_idx, "best_val": best_val, "rmse": rmse, "mape": mape}, f, indent=2)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("SPIKAN XJTU runner (separate)")
    p.add_argument('--data', type=str, default='XJTU')
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--normalization_method', type=str, default='min-max')
    p.add_argument('--log_dir', type=str, default=None)

    # scheduler
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--early_stop', type=int, default=20)
    p.add_argument('--warmup_epochs', type=int, default=30)
    p.add_argument('--warmup_lr', type=float, default=0.002)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--final_lr', type=float, default=0.0002)
    p.add_argument('--lr_F', type=float, default=0.001)

    # model
    p.add_argument('--F_layers_num', type=int, default=3)
    p.add_argument('--F_hidden_dim', type=int, default=60)

    # loss
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--beta', type=float, default=20.0)
    p.add_argument('--beta_warmup_epochs', type=int, default=50)  # NEW

    # saving/batches
    p.add_argument('--save_root', type=str, default='runs_spikan')
    p.add_argument('--experiments', type=int, default=10)
    p.add_argument('--batches', type=str, nargs='*',
                   default=['2C','3C','R2.5','R3','RW','satellite'],
                   help="subset of batches to run, e.g. --batches 2C RW")
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

    for b in args.batches:
        for e in range(1, args.experiments + 1):
            train_one_run(args, b, e)

if __name__ == "__main__":
    main()
