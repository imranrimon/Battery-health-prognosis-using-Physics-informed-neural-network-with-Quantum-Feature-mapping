# main_XJTU_time.py (single-batch, single-experiment, no early stop)
import argparse, os, time, json, csv, datetime, torch
from dataloader.dataloader import XJTUdata
from Model.Model import PINN

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES','0')

def makedirs(p): os.makedirs(p, exist_ok=True)

def write_json(obj, path):
    makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_row_csv(path, header, row_dict):
    makedirs(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(row_dict)

def load_data(args, small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list, test_list = [], []
    for file in os.listdir(root):
        if args.batch in file:
            (test_list if ('4' in file or '8' in file) else train_list).append(os.path.join(root, file))
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader  = data.read_all(specific_path_list=test_list)
    return {'train': train_loader['train_2'],
            'valid': train_loader['valid_2'],
            'test' : test_loader['test_3']}

def disable_early_stop(pinn, args):
    """Make sure training runs full epochs even if the model has internal patience."""
    # keep args consistent
    args.early_stop = max(getattr(args, 'early_stop', 0), 10**9)
    # common attribute names inside PINN
    for name in ['early_stop', 'patience', 'patience_epochs', 'es_patience']:
        if hasattr(pinn, name):
            try: setattr(pinn, name, 10**9)
            except Exception: pass
    # sometimes kept inside scheduler or a trainer dict
    if hasattr(pinn, 'scheduler') and hasattr(pinn.scheduler, 'patience'):
        try: pinn.scheduler.patience = 10**9
        except Exception: pass

def get_args():
    p = argparse.ArgumentParser('XJTU timing (single batch/experiment, no ES)')
    # choose ONE batch
    p.add_argument('--batch', type=str, default='2C',
                   choices=['2C','3C','R2.5','R3','RW','satellite'])
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--normalization_method', type=str, default='min-max')

    # training config
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--warmup_epochs', type=int, default=30)
    p.add_argument('--warmup_lr', type=float, default=0.002)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--final_lr', type=float, default=0.0002)
    p.add_argument('--lr_F', type=float, default=0.001)

    # model/loss (pass-through to PINN)
    p.add_argument('--F_layers_num', type=int, default=3)
    p.add_argument('--F_hidden_dim', type=int, default=60)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--beta',  type=float, default=0.2)

    # io
    p.add_argument('--save_folder', type=str, default='results_time/XJTU')  # no spaces
    p.add_argument('--log_file', type=str, default='logging.txt')
    p.add_argument('--small_sample', type=int, default=None)
    # run id (optional)
    p.add_argument('--experiment', type=int, default=1)
    return p.parse_args()

def main():
    args = get_args()

    base_root = os.path.abspath(args.save_folder).replace(' ', '_')
    makedirs(base_root)

    # file layout: <base>/<batch-idx>-<batch-idx>/Experiment<id> (kept for compatibility)
    batch_order = ['2C','3C','R2.5','R3','RW','satellite']
    batch_idx = batch_order.index(args.batch)
    exp_dir = os.path.join(base_root, f'{batch_idx}-{batch_idx}', f'Experiment{args.experiment}')
    makedirs(exp_dir)

    # tell PINN where to save its usual artifacts
    args.save_folder = exp_dir
    args.log_dir = args.log_file

    # load data & model
    loaders = load_data(args, small_sample=args.small_sample)
    pinn = PINN(args)
    disable_early_stop(pinn, args)  # <-- force no early stop

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    except Exception:
        gpu_name = 'unknown'

    # timing
    t0 = time.perf_counter()
    start_iso = datetime.datetime.now().isoformat(timespec='seconds')

    pinn.Train(trainloader=loaders['train'],
               validloader=loaders['valid'],
               testloader=loaders['test'])

    secs = time.perf_counter() - t0
    end_iso = datetime.datetime.now().isoformat(timespec='seconds')
    sec_per_epoch = secs / max(1, args.epochs)

    # records
    header = ['dataset','batch_idx','batch_tag','experiment',
              'epochs','train_seconds','sec_per_epoch',
              'start_iso','end_iso','save_folder','log_path','device','gpu_name']
    row = {
        'dataset': 'XJTU',
        'batch_idx': batch_idx,
        'batch_tag': args.batch,
        'experiment': args.experiment,
        'epochs': args.epochs,
        'train_seconds': secs,
        'sec_per_epoch': sec_per_epoch,
        'start_iso': start_iso,
        'end_iso': end_iso,
        'save_folder': exp_dir,
        'log_path': os.path.join(exp_dir, args.log_file),
        'device': device,
        'gpu_name': gpu_name
    }

    # per-experiment
    write_json(row, os.path.join(exp_dir, 'train_time.json'))
    write_row_csv(os.path.join(exp_dir, 'train_time.csv'), header, row)
    # global index
    write_row_csv(os.path.join(base_root, 'train_time_index.csv'), header, row)

    print(f"[OK] {args.batch} exp {args.experiment}: {secs:.1f}s "
          f"({sec_per_epoch:.3f}s/epoch) on {device} [{gpu_name}]")

if __name__ == '__main__':
    main()
