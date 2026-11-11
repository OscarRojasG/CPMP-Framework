# Training of the policy network

Self-contained code of the two stages that produced the policy of the paper
(`models/sl.pth` after imitation, `models/rl.pth` after REINFORCE).

`src/` is the version of the framework with which the reported model was trained. It predates the
reorganization of the top-level `src/` (module names differ, e.g.
`models/actions/cpmp_transformer_V2.py` here is `models/actions/default.py` there); the network is
the same and the checkpoints are interchangeable. Use this folder to train and the top-level `src/`
to evaluate.

| File | Purpose |
|---|---|
| `train_sl.py` | Imitation of FRG on the supervised bank |
| `train_rl.py` | REINFORCE refinement starting from the imitation model |
| `src/` | Framework code used by both scripts (`training/sl/training.py`, `training/rl/training_v2.py`, models, generators, adapters) |
| `original_run/reinforce_all.py` | Launch script of the REINFORCE run reported in the paper |
| `logs/rl_job17551.out` | Log of that run (the validation curve of the paper's training figure); the absolute path of the saved model was shortened to its file name |

## Requirements

PyTorch (the reported run used 2.6.0 with CUDA 12.4), `numpy`, `h5py`, `scikit-learn` and
`python-dotenv`. Both
scripts read and write everything in the folder given by the environment variable `WORK_PATH`,
and can be run from any directory. They put `training/src/` first in the import path, so they work
whether or not the top-level package is installed (`pip install -e .`).

## Stage 1: imitation

Recipe (Table 3 of the paper): 304,000 / 76,000 training / validation states (80/20 split
stratified by cost), 100 epochs, batch size 128, AdamW with learning rate 1e-4 and weight decay
1e-4, dropout 0.1. The checkpoint with the lowest validation loss is kept.

```bash
export WORK_PATH=/path/to/workdir        # must contain actions_sl.data (see "Datasets" in the main README)
python -u train_sl.py --seed 42
```

Output in `WORK_PATH`: `sl.pth` / `sl.json` (best model), `sl_history.csv` (losses
and accuracy per epoch) and `sl_last.pt` (state to resume an interrupted run).

## Stage 2: REINFORCE

Recipe (Table 3 of the paper): 19 bay sizes; a fixed pool of 128,000 training instances per size
and 40 validation instances per size (760 in total) drawn with the CV generator; 8 instances per
size and K = 16 trajectories per instance in each update; KL weight 0.1 with respect to the
reference policy; advantages clipped to [-4, 4]; gradient clipping 1.0; AdamW with learning rate
1e-4 and no weight decay; step limit 60 / 100 / 150 for N ≤ 20 / ≤ 40 / > 40 containers; greedy
evaluation on the validation pool every 10 updates; at most 10,000 updates, stopping after more
than 50 evaluations without improvement.

```bash
export WORK_PATH=/path/to/workdir        # must contain sl.pth and sl.json (copy them from models/ and models/hyperparameters/)
python -u train_rl.py --seed 42
```

Output in `WORK_PATH`: `rl.pth` / `rl.json` (best model on the validation pool),
`rl_dev.csv` (validation curve), `rl_updates.csv` (telemetry per update) and
`rl_last.pt` (state to resume).

The reported run used a single NVIDIA V100 GPU and took 19.2 h: it stopped at update 2,460 with a
best validation score of 25.65 moves (`logs/rl_job17551.out`). Runs on GPU are not reproducible bit
by bit, so a new run yields different weights with a validation curve close to the reported one.

`train_rl.py` follows `original_run/reinforce_all.py` (same recipe, same pools and same order of
operations) and adds checkpointing and the CSV logs.

Note on versions. The reported run was launched with an earlier revision of the REINFORCE loop,
whose log prints an adaptive sampling temperature (`Temp`) and stops by a "rollback" limit. In that
run the temperature stayed at 1.00 in all updates and the rollback limit was 0, so it behaved as
`src/training/rl/training_v2.py` does: sampling at temperature 1 and stopping once the patience is
exhausted, keeping the best model. `training_v2.py` also adds an initial greedy evaluation of the
imitation model, which consumes no random numbers.
