# CPMP-Framework

Code, trained models and results for the paper **"A Size-Independent Self-Attention Model for the Container Pre-Marshalling Problem"** (I. Araya, F. Astudillo, O. Rojas, I. Guerrero).

A self-attention encoder-decoder with no parameter tied to the width or height of the bay, so that a single network serves every training size of the Container Pre-Marshalling Problem (CPMP). The action policy imitates the FRG heuristic, is refined with REINFORCE, and is combined with a cost network that estimates the remaining moves to guide beam search.

## Installation

Requires Python 3.12+ on Linux (the FRG heuristic is shipped as a Linux x86-64 binary in `src/frg`).

```bash
git clone https://github.com/OscarRojasG/CPMP-Framework.git
cd CPMP-Framework
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

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

Place them in `data/`. They can be regenerated with `scripts/data/actions.py` and `scripts/data/cost.py`.

## Reproducing the experiments

Run the scripts from the repository root; each one has its parameters at the top of the file.

```bash
python scripts/benchmarking/greedy.py     # Greedy on CV
python scripts/benchmarking/bsg.py        # BSG on CV
python scripts/benchmarking/bs_cost.py    # BS+Cost on CV
python scripts/benchmarking/g_bscost.py   # BS+Cost on G1/G2/G3 (set DATASET at the top)
```

`g_greedy.py` and `g_bsg.py` run Greedy and BSG on G1–G3. Reinforcement refinement is in `scripts/training/rl.py`.

Results are written to `experiments/` as one CSV per configuration (instance, solved, moves and, for CV, time). `notebooks/experiments.ipynb` aggregates them into tables.

## Citation

```bibtex
@article{araya2026cpmp,
  title  = {A Size-Independent Self-Attention Model for the Container Pre-Marshalling Problem},
  author = {Araya, Ignacio and Astudillo, Felipe and Rojas, Oscar and Guerrero, Ian},
  year   = {2026},
  note   = {Under review}
}
```

## License

Code released under the [MIT License](LICENSE).
