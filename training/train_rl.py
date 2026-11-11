"""REINFORCE refinement with the recipe of the run reported in the paper (Table 3, Section 4.3).

Uses `sample_pomo_rollouts`, `compute_pomo_advantages`, `update_model_pomo` and
`evaluate_dev_greedy` of `src/training/rl/training_v2.py` unchanged. The loop is the one of
`train_pomo_rl()` in that module, in the same order (including the initial evaluation) and with
the same calls to the RNG; it only adds writing to disk:

  <name>.pth / .json      best model on the validation pool (replaced on every improvement)
  <name>_updates.csv      telemetry of every update
  <name>_dev.csv          every evaluation on the 760 validation instances (update 0 = model
                          after imitation)
  <name>_last.pt          full state after every evaluation, to resume an interrupted job
  <name>.done             written at the end

The pools (760 validation instances, 128,000 x 19 training instances) always come from
`--pool-seed` 42; `--seed` controls the randomness of training. The model of the paper
corresponds to `--seed 42` (with seed == pool-seed the RNG is not re-seeded after building the
pools, as in the original `train()`).

Requires the environment variable WORK_PATH (folder with the imitation model `--sl-name`;
everything is written there).
"""
import argparse
import json
import os
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--pool-seed", type=int, default=42)
ap.add_argument("--sl-name", default="sl")
ap.add_argument("--name", default="rl")
ap.add_argument("--threads", type=int, default=min(20, os.cpu_count() or 1),
                help="torch threads (default: up to 20 cores)")
# Valores de la corrida original; cambiarlos sólo para pruebas cortas.
ap.add_argument("--updates", type=int, default=10000)
ap.add_argument("--train-size", type=int, default=128000)
ap.add_argument("--test-size", type=int, default=40)
ap.add_argument("--eval-interval", type=int, default=10)
args = ap.parse_args()

sys.path.insert(0, args.src)
import numpy as np
import torch
from cpmp.layout import Layout
from training.sl.training import load_model
from training.rl.training_v2 import (POMOConfig, compute_pomo_advantages, evaluate_dev_greedy,
                                     sample_pomo_rollouts, update_model_pomo)
from generation.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV2, StackFeaturesAdapterV1
from models.actions.cpmp_transformer_V2 import CPMPTransformer
from instances.generators import CVGenerator

COMBOS = [(3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5),
          (4, 6), (5, 6), (6, 6), (7, 6),
          (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7),
          (6, 8), (10, 8)]


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_best(model, work, name):
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
    P = lambda suffix: os.path.join(work, args.name + suffix)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.is_available()} "
          f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} | seed {args.seed} "
          f"| pool_seed {args.pool_seed}", flush=True)

    model = load_model(CPMPTransformer, args.sl_name).to(device)
    ref_model = load_model(CPMPTransformer, args.sl_name).to(device)
    cfg = POMOConfig(updates=args.updates, k_rollouts=16, instances_per_combo=8, kl_coef=0.1,
                     adv_clip=4.0, grad_clip=1.0, minibatch_size=2048,
                     eval_interval=args.eval_interval, patience=50)
    input_adapter_config = (EnrichedLayoutAdapter, Layout4DAdapterV2, StackFeaturesAdapterV1, 10, 8)

    # --- train() de training_v2: generadores, topes y pools (siempre con pool_seed) ---
    generators = [CVGenerator(S=S, H=H, seed=args.pool_seed) for S, H in COMBOS]
    max_steps_dict = {}
    for gen in generators:
        N = gen.S * (gen.H - 2)
        max_steps_dict[(gen.S, gen.H)] = 60 if N <= 20 else (100 if N <= 40 else 150)
    random.seed(args.pool_seed)
    np.random.seed(args.pool_seed)
    torch.manual_seed(args.pool_seed)
    print("Generando pool estático de instancias de entrenamiento y validación...", flush=True)
    dev_instances, train_instances_dict = [], {}
    for gen in generators:
        dev_instances.extend([Layout(inst, gen.H) for inst in gen.generate_instances(args.test_size)])
        train_instances_dict[gen] = [Layout(inst, gen.H) for inst in gen.generate_instances(args.train_size)]
    if args.seed != args.pool_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    print(f"Pools: {len(dev_instances)} de validación, {args.train_size} de entrenamiento por combo", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # --- train_pomo_rl() con guardado continuo ---
    start, best, no_improve, best_state = 1, float("inf"), 0, None
    if os.path.exists(P("_last.pt")):
        ck = torch.load(P("_last.pt"), map_location="cpu", weights_only=False)  # RNG debe quedar en CPU
        # Rehacer el mejor modelo desde el checkpoint (antes de restaurar el RNG: el constructor
        # llama a torch.manual_seed). Cubre una caída entre guardar un .pth mejor y el _last.pt.
        if ck.get("best_state") is not None:
            best_state = ck["best_state"]
            bm = load_model(CPMPTransformer, args.sl_name)
            bm.load_state_dict(best_state)
            save_best(bm, work, args.name)
        model.load_state_dict(ck["model"])
        ref_model.load_state_dict(ck["ref_model"])
        optimizer.load_state_dict(ck["optimizer"])
        random.setstate(ck["rng_python"])
        np.random.set_state(ck["rng_numpy"])
        torch.set_rng_state(ck["rng_torch"])
        if torch.cuda.is_available() and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        start, best, no_improve = ck["update"] + 1, ck["best"], ck["no_improve"]
        truncate_csv(P("_updates.csv"), ck["update"])
        truncate_csv(P("_dev.csv"), ck["update"])
        print(f"Reanudando desde el update {start} (mejor {best:.2f}, sin mejora {no_improve})", flush=True)
    else:
        with open(P("_updates.csv"), "w") as f:
            f.write("update,pg_loss,entropy,kl,solve_rate,avg_steps,norm_ent,steps_per_s\n")
        with open(P("_dev.csv"), "w") as f:
            f.write("update,dev_score,solved,total,best,no_improve,saved\n")
        dev, solved, total = evaluate_dev_greedy(model, dev_instances, max_steps_dict, input_adapter_config, device)
        print(f"   >>> 🧪 Evaluación Inicial: {dev:.2f} pasos | Resueltas: {solved}/{total} ({100*solved/total:.1f}%)", flush=True)
        with open(P("_dev.csv"), "a") as f:
            f.write(f"0,{dev:.4f},{solved},{total},,0,0\n")

    print(f"Iniciando POMO RL por {cfg.updates} updates...", flush=True)
    update = start - 1
    stopped = no_improve > cfg.patience          # ya se había cumplido la parada antes de la caída
    for update in range(start, cfg.updates + 1):
        if stopped:
            update -= 1
            break
        t0 = time.time()
        instances = []
        for gen in generators:
            instances.extend(random.sample(train_instances_dict[gen], cfg.instances_per_combo))
        states_h, actions_h, returns, combo_ids, rollout_idx, is_solved, norm_ent = sample_pomo_rollouts(
            model, instances, cfg.k_rollouts, max_steps_dict, input_adapter_config, device)
        adv = compute_pomo_advantages(returns, combo_ids, rollout_idx, cfg.k_rollouts, cfg.adv_clip, device)
        pg, ent, kl = update_model_pomo(model, ref_model, optimizer, states_h, actions_h, adv,
                                        cfg.minibatch_size, cfg.kl_coef, cfg.grad_clip, device)
        solve_rate = is_solved.float().mean().item() * 100
        avg_steps = -returns[is_solved].mean().item() if solve_rate > 0 else 0.0
        sps = len(actions_h) / max(1e-4, time.time() - t0)
        print(f"Update {update:05d} | PG: {pg:+.3f} | Ent: {ent:.3f} | KL: {kl:.3f} | Solve: {solve_rate:3.0f}% "
              f"| AvgSteps: {avg_steps:.1f} | NormEnt: {norm_ent:.2f} | {sps:.0f} steps/s", flush=True)
        with open(P("_updates.csv"), "a") as f:
            f.write(f"{update},{pg:.5f},{ent:.5f},{kl:.5f},{solve_rate:.2f},{avg_steps:.3f},{norm_ent:.4f},{sps:.0f}\n")

        if update % cfg.eval_interval == 0:
            dev, solved, total = evaluate_dev_greedy(model, dev_instances, max_steps_dict, input_adapter_config, device)
            print(f"   >>> 🧪 Evaluación Dev: {dev:.2f} pasos | Resueltas: {solved}/{total} "
                  f"({100*solved/total:.1f}%) | (Mejor: {min(dev, best):.2f})", flush=True)
            saved = dev < best
            if saved:
                best, no_improve = dev, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_best(model, work, args.name)
                ref_model.load_state_dict(model.state_dict())
                print("   >>> 🏆 Nuevo mejor modelo guardado; ref_model actualizado.", flush=True)
            else:
                no_improve += 1
                print(f"   >>> ⚠️ Sin mejora ({no_improve}/{cfg.patience}).", flush=True)
            with open(P("_dev.csv"), "a") as f:
                f.write(f"{update},{dev:.4f},{solved},{total},{best:.4f},{no_improve},{int(saved)}\n")
            atomic_save({
                "model": model.state_dict(), "ref_model": ref_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                "rng_torch": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "update": update, "best": best, "no_improve": no_improve, "best_state": best_state,
            }, P("_last.pt"))
            if no_improve > cfg.patience:
                print(f"   >>> 🛑 Detenido: {cfg.patience} evaluaciones sin mejora.", flush=True)
                stopped = True
                break

    open(P(".done"), "w").write(f"best_dev={best:.4f} last_update={update} stopped_by_patience={stopped}\n")
    print(f"✅ REINFORCE terminado: mejor {best:.2f}, último update {update}.", flush=True)


if __name__ == "__main__":
    main()
