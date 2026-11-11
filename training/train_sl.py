"""Imitation stage (supervised learning on the FRG-labelled bank), Table 3 and Section 4.2 of the paper.

Uses `train_epoch`, `val_epoch`, `generate_sets` and `config_training` of `src/training/sl/training.py`
unchanged. The epoch loop is the one of `_train()` in that module (same optimizer, constant
learning rate with the defaults of LRConfig, GradScaler), plus writing to disk after every epoch:

  <name>.pth / .json     best model so far (lowest validation loss)
  <name>_history.csv     losses and accuracy per epoch
  <name>_last.pt         full state to resume (model, optimizer, scheduler, scaler, RNG)
  <name>.done            written when all epochs are finished

If <name>_last.pt exists the run resumes from the next epoch. `--seed` controls the initialization
and the order of the data; the 80/20 split stratified by cost is fixed by `--split-seed` (42).
Requires the environment variable WORK_PATH (folder that contains the bank `actions_sl.data`;
everything is written there). The model of the paper corresponds to `--seed 42`.
"""
import argparse
import os
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"),
                help="folder with the training code (default: training/src)")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--split-seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--data", default="actions_sl.data")
ap.add_argument("--name", default="sl")
ap.add_argument("--threads", type=int, default=min(20, os.cpu_count() or 1),
                help="torch threads and DataLoader workers (default: up to 20 cores)")
ap.add_argument("--max-size", type=int, default=None, help="use only the first N samples (for short tests)")
args = ap.parse_args()

# El código de src usa os.cpu_count() para los workers del DataLoader y los hilos de torch; en
# un nodo compartido se limita a --threads.
os.cpu_count = lambda: args.threads
sys.path.insert(0, args.src)

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from preprocessing.dataset import load_dataset
from training.sl.training import LRConfig, config_training, generate_sets, train_epoch, val_epoch
from training.metrics import Accuracy, CrossEntropyLoss
from models.actions.cpmp_transformer_V2 import CPMPTransformer

ARCH = dict(H_dim=8, C_dim=3, X_dim=3, d_model=64, nhead=4, num_layers=4,
            ff_dim_multiplier=2, dropout=0.1)
BATCH, LR, WD = 128, 1e-4, 1e-4


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_best(model, work, name):
    import json
    with open(os.path.join(work, name + ".json.tmp"), "w") as f:
        json.dump(model.hyperparams, f, indent=4)
    os.replace(os.path.join(work, name + ".json.tmp"), os.path.join(work, name + ".json"))
    atomic_save(model.state_dict(), os.path.join(work, name + ".pth"))


def truncate_csv(path, last_index):
    """Deja la cabecera y las filas con índice (primera columna) <= last_index."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = f.readlines()
    keep = lines[:1] + [l for l in lines[1:] if l.split(",")[0].strip() and int(l.split(",")[0]) <= last_index]
    with open(path + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(path + ".tmp", path)


def main():
    work = os.environ["WORK_PATH"]
    last_path = os.path.join(work, args.name + "_last.pt")
    hist_path = os.path.join(work, args.name + "_history.csv")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Transformer.__init__ (models/transformer.py:7) hace torch.manual_seed(42) antes de crear las
    # capas, así que toda red parte de los pesos de la semilla 42. Se neutraliza esa llamada sólo
    # durante la construcción para que --seed controle la inicialización. Con --seed 42 los pesos
    # son idénticos a los del constructor original (mismo estado del RNG).
    _manual_seed = torch.manual_seed
    torch.manual_seed = lambda *a, **k: None
    try:
        model = CPMPTransformer(**ARCH)
    finally:
        torch.manual_seed = _manual_seed

    data = load_dataset(args.data, args.max_size)
    n = len(data)
    train_size = int(round(0.8 * n))
    test_size = n - train_size
    print(f"seed={args.seed} split_seed={args.split_seed} train={train_size} val={test_size}", flush=True)

    device = config_training(model, args.seed)   # random/torch seed, dispositivo, hilos
    train_set, test_set = generate_sets(data, train_size, test_size, args.split_seed)

    # Igual que _train() de training/sl/training.py
    pin = device.type in ("cuda", "mps")
    train_loader = DataLoader(train_set, batch_size=BATCH, num_workers=os.cpu_count(), pin_memory=pin, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH, num_workers=os.cpu_count(), pin_memory=pin)
    lr_cfg = LRConfig(start=LR)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_cfg.start, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=lr_cfg.factor, patience=lr_cfg.patience, min_lr=lr_cfg.min)
    scaler = GradScaler(device.type)
    loss_functions, metrics = [CrossEntropyLoss()], [[Accuracy()]]

    start_epoch, best_val, best_epoch, best_state = 1, float("inf"), 0, None
    if os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)  # RNG debe quedar en CPU
        # El mejor modelo se rehace desde el checkpoint (antes de restaurar el RNG, porque el
        # constructor llama a torch.manual_seed): si el job murió entre guardar un .pth mejor y el
        # _last.pt, el .pth en disco no correspondería a best_val.
        if ck.get("best_state") is not None:
            best_model = CPMPTransformer(**ARCH)
            best_model.load_state_dict(ck["best_state"])
            save_best(best_model, work, args.name)
            best_state = ck["best_state"]
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        random.setstate(ck["rng_python"])
        np.random.set_state(ck["rng_numpy"])
        torch.set_rng_state(ck["rng_torch"])
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        start_epoch, best_val, best_epoch = ck["epoch"] + 1, ck["best_val"], ck["best_epoch"]
        truncate_csv(hist_path, ck["epoch"])
        print(f"Reanudando desde la época {start_epoch} (mejor val {best_val:.4f} en la época {best_epoch})", flush=True)
    elif not os.path.exists(hist_path):
        with open(hist_path, "w") as f:
            f.write("epoch,train_ce,val_ce,val_acc,lr,seconds,best\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        (tr_loss,), _ = train_epoch(model, train_loader, optimizer, loss_functions, metrics, device, scaler)
        (va_loss,), ((va_acc,),) = val_epoch(model, test_loader, loss_functions, metrics, device)
        improved = va_loss < best_val
        if improved:
            best_val, best_epoch = va_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_best(model, work, args.name)
        scheduler.step(va_loss)
        secs = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        with open(hist_path, "a") as f:
            f.write(f"{epoch},{tr_loss:.6f},{va_loss:.6f},{va_acc:.4f},{lr:.2e},{secs:.1f},{int(improved)}\n")
        atomic_save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch, "best_state": best_state,
        }, last_path)
        print(f"Epoch {epoch}/{args.epochs} | train CE {tr_loss:.4f} | val CE {va_loss:.4f} | "
              f"val acc {va_acc:.2f}% | {secs:.0f} s{' | mejor, guardado' if improved else ''}", flush=True)

    open(os.path.join(work, args.name + ".done"), "w").write(
        f"best_epoch={best_epoch} best_val_ce={best_val:.6f}\n")
    print(f"Imitación terminada: mejor época {best_epoch} (val CE {best_val:.4f})", flush=True)


if __name__ == "__main__":
    main()
