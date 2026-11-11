# CPMP-Framework

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24095036.svg)](https://doi.org/10.5281/zenodo.24095036)
[![Datasets DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24093623.svg)](https://doi.org/10.5281/zenodo.24093623)

Code, trained models and results for the paper **"A Size-Independent Self-Attention Model for the Container Pre-Marshalling Problem"** (I. Araya, F. Astudillo, O. Rojas, I. Guerrero).

A self-attention encoder-decoder with no parameter tied to the width or height of the bay, so that a single network serves every training size of the Container Pre-Marshalling Problem (CPMP). The action policy imitates the FRG heuristic, is refined with REINFORCE, and is combined with a cost network that estimates the remaining moves to guide beam search.

## Installation

Requires Python 3.12+. The evaluation scripts, the notebooks and the training code run on Linux and macOS, with or without a GPU.

```bash
git clone https://github.com/OscarRojasG/CPMP-Framework.git
cd CPMP-Framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is the pinned environment of the experiments (Linux with CUDA). On another platform, install PyTorch for your system and then the remaining packages:

```bash
pip install torch numpy scipy h5py pandas scikit-learn matplotlib python-dotenv jupyter
pip install -e .
```

The FRG heuristic called from Python (`FRGSolver` and `BSGFRGSolver`, needed only to regenerate the training datasets) is shipped as a Linux x86-64 binary in `src/frg`. The heuristic baselines of the paper were run with a fork of the authors' C++ solver; [`frg_solver/`](frg_solver/README.md) links to it and explains how to build it on Linux and macOS. Its command line takes one more argument than `src/frg`, so it is a standalone program and not a replacement for that binary.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Framework used in every evaluation: layout, models, solvers, evaluators, instance generators and data adapters |
| `models/` | Trained networks of the paper and their hyperparameters |
| `instances/benchmarks/` | Benchmark instances (CV set and the DLTS classes G1–G3) |
| `scripts/benchmarking/`, `scripts/data/` | Evaluation on the benchmark sets and generation of the training datasets |
| `training/` | Self-contained code of the imitation and REINFORCE stages, with the launch script and the log of the run reported in the paper ([`training/README.md`](training/README.md)) |
| `frg_solver/` | How to obtain and build the C++ solver of FRG and BS-FRG, the heuristic baselines ([`frg_solver/README.md`](frg_solver/README.md)) |
| `experiments/` | Per-instance results behind the tables and figures of the paper ([`experiments/README.md`](experiments/README.md)) |
| `notebooks/` | `experiments.ipynb` aggregates the result files into tables; `dlts.ipynb` is a walkthrough of the DLTS port |

## Solving strategies

| Paper | Solver class | Description |
|---|---|---|
| Greedy | `ModelSolver` | Follows the action policy greedily |
| BSG | `BSGModelSolver` | Beam search guided by the action policy |
| BS+Cost | `BSGCostPredictorSolver` | Beam search ranked by the cost network instead of rollouts |
| BS-FRG | `BSGFRGSolver` | Beam search guided by the FRG heuristic (baseline) |
| DLTS port | `DLTSSolver` | Reimplementation of the DLTS search running with our networks |

All solvers live in `src/solvers/`.

## Trained models

| Model | Description |
|---|---|
| `models/sl.pth` | Action network after imitation (behavioral cloning of FRG) |
| `models/rl.pth` | Action network refined with REINFORCE — the final model used in every experiment |
| `models/cost.pth` | Cost network |
| `models/ablation_sf.pth` | Ablation without stack features, trained up to the imitation stage (`models/actions/no_stack_features.py`) |
| `models/experts/rl_{T}-{S}.pth` | 19 per-size experts, one per CV group, refined with REINFORCE from the imitation checkpoint |

Hyperparameters of each model are in `models/hyperparameters/`. To load a model:

```python
from training.common import load_model
from models.actions.default import CPMPTransformer

model = load_model(CPMPTransformer, "rl")               # or "sl", "experts/rl_5-5"
```

## Benchmark instances

| Folder | Description |
|---|---|
| `instances/benchmarks/cvs/` | Caserta–Voß (CV) set: 19 groups `T-S` (initial tiers – stacks), 40 instances each, maximum height `H = T + 2` |
| `instances/benchmarks/g1/`, `g2/`, `g3/` | Instances published with DLTS (Hottung et al., 2020): classes G1 (unique priorities), G2 and G3 (each priority shared by two or three containers), 250 test instances per size. Folders `5-5`, `5-7` and `5-10` correspond to sizes 5×7, 7×7 and 10×7; the `*_sol` folders hold the reference solutions |

## Datasets

The training datasets (HDF5, 380,000 samples each: 20,000 per CV group) are archived separately on Zenodo: [10.5281/zenodo.24093623](https://doi.org/10.5281/zenodo.24093623).

- `actions_sl.data`: imitation dataset for the action network.
- `cost.data`: training dataset for the cost network.

Place them in `data/`. They can be regenerated with `scripts/data/actions.py` and `scripts/data/cost.py` (Linux only, see Installation). The target of the cost network is `ln(c_min + 1)`, as in the paper; in the archived `cost.data` the field `C` also holds `c_min + 1`, whereas a regenerated file stores `c_min` there (the field is not used as a target).

## Reproducing the experiments

Run the scripts from the repository root; each one has its parameters at the top of the file.

```bash
python scripts/benchmarking/greedy.py     # Greedy on CV
python scripts/benchmarking/bsg.py        # BSG on CV
python scripts/benchmarking/bs_cost.py    # BS+Cost on CV
python scripts/benchmarking/g_bscost.py   # BS+Cost on G1/G2/G3 (set DATASET at the top)
```

`g_greedy.py` and `g_bsg.py` run Greedy and BSG on G1–G3.

Results are written to `experiments/` as one CSV per configuration (instance, solved, moves and, for CV, time), replacing the file of the same name shipped with the repository. `notebooks/experiments.ipynb` aggregates them into tables.

Average moves over the 19 CV groups (760 instances), as reported in the paper:

| | FRG | Greedy | BS+Cost 128 | BS+Cost 256 | BS+Cost 512 | BSG 16 | BSG 32 | BS-FRG 10⁴ |
|---|---|---|---|---|---|---|---|---|
| Moves | 31.29 | 25.72 | 24.31 | 24.25 | 24.20 | 24.31 | 24.26 | 24.99 |
| Time per instance (s) | <0.0001 | 0.058 | 22.00 | 43.07 | 85.26 | 26.64 | 88.88 | 23.11 |

Times were measured on one core of a server with two Intel Xeon Gold 6148 processors. Moves do not depend on the hardware for Greedy and BS+Cost; BSG breaks ties between integer rollout lengths, and its moves can differ slightly across platforms.

The results of the baselines (FRG, BS-FRG), of the DLTS port, of the analysis of the cost network and of the statistical tests are in subfolders of `experiments/`, described in [`experiments/README.md`](experiments/README.md).

## Training

The policy was obtained in two stages: imitation of FRG and refinement with REINFORCE. The code of both stages, the recipe (Table 3 of the paper), the launch script and the log of the reported run are in [`training/`](training/README.md):

```bash
export WORK_PATH=/path/to/workdir     # holds actions_sl.data (stage 1) and sl.pth / sl.json (stage 2)
python -u training/train_sl.py --seed 42
python -u training/train_rl.py --seed 42
```

`training/src/` is the revision of the framework with which the reported model was trained. Use it to train and the top-level `src/` to evaluate: the top-level models cache the stack embeddings without gradient, which is what inference needs, so they are not meant for training. The network is the same and the checkpoints are interchangeable.

## Citation

```bibtex
@article{araya2026cpmp,
  title  = {A Size-Independent Self-Attention Model for the Container Pre-Marshalling Problem},
  author = {Araya, Ignacio and Astudillo, Felipe and Rojas, Oscar and Guerrero, Ian},
  year   = {2026},
  note   = {Under review}
}
```

The code and trained models of the paper are archived on Zenodo: [10.5281/zenodo.24095036](https://doi.org/10.5281/zenodo.24095036) resolves to the latest release (v1.0.0 is [10.5281/zenodo.24095037](https://doi.org/10.5281/zenodo.24095037)). The training datasets are at [10.5281/zenodo.24093623](https://doi.org/10.5281/zenodo.24093623).

## License

Code released under the [MIT License](LICENSE).
