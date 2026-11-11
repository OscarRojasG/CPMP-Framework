# Results behind the tables and figures of the paper

All runs were executed on a server with two Intel Xeon Gold 6148 processors. File and column
names of the baseline and analysis outputs are in Spanish (`pasos` = moves, `resueltas` = solved,
`grupo` = group, `movs` = mean moves, `seg` / `t_s` = seconds, `nodos` = nodes).

## Our methods on the benchmark sets

One CSV per solver, beam width, benchmark set and group, in this folder, written by
`scripts/benchmarking/*.py` (columns `Instance, Solved, Steps` and, on the CV set, `Time`; the limit
is 100 moves).

| Files | Method |
|---|---|
| `ModelSolver_<set>_<group>.csv` | Greedy (the policy alone) |
| `BSGCostPredictorSolver_w<width>_<set>_<group>.csv` | BS+Cost, beam search guided by the cost network |
| `BSGModelSolver_w<width>_<set>_<group>.csv` | BSG, beam search with greedy rollouts of the policy |

`<set>` is `cvs` (CV set) or `g1`, `g2`, `g3` (DLTS instances, test split; folders `5-5`, `5-7` and
`5-10` are the sizes 5×7, 7×7 and 10×7).

The `cvs` files give the columns Greedy, BS+Cost and BSG of the main table on the CV set and the
points of the quality-versus-time figure; the `g1`–`g3` files with w = 32 give the column BS+Cost of
the table on repeated priorities (the same moves as `DLTS_sets/tabla_L.csv`).

## `FRG_BSFRG/` — heuristic baselines on the CV set

Runs of the C++ solver described in [`../frg_solver`](../frg_solver/README.md), random seed 2.

| File | Content |
|---|---|
| `frg_baselines_cluster.json` | FRG with the unblocking assignment of destination stacks, 760 instances. Column FRG of the main table (average 31.29) |
| `frg_baselines_cluster_k3_bs.json` | BS-FRG with 1,000 and 10,000 candidates, k = 3. Moves of BS-FRG (24.99 with 10,000 candidates) |
| `tabla_frg_bsfrg_cluster_k3.csv` | Per-group summary of the previous file |
| `one_core_times/` | The same BS-FRG runs repeated on a single dedicated core, one file per group and beam size; they give the reported times |
| `tabla_tiempos_c1.csv` | Per-group summary of the single-core times (`*_t_c1`; average 23.11 s for 10,000 candidates) |

Each JSON holds the solver settings (`solver`), a summary per group (`por_grupo`) and overall
(`summary`), and the per-instance records (`raw`).

## `DLTS_sets/` — instances of Hottung et al. (2020), classes G1–G3

Test and validation splits of sizes 5×7, 7×7 and 10×7 (stacks × maximum height), 250 instances
per size and class.

| File | Content |
|---|---|
| `paper_L_t100_por_instancia.json`, `tabla_L.csv` | Test split: Greedy, BS+Cost with w = 32 and the port of DLTS with 8,192 nodes. Column BS+Cost of the table on repeated priorities |
| `paper_L_c1_w32_por_instancia.csv` | Test split: BS+Cost with w = 32 run again on one dedicated core (same moves, instance by instance, as `paper_L_t100_por_instancia.json`). Times of BS+Cost in the table on repeated priorities: mean of `time_s` over the 750 instances of each size (2.0 / 6.1 / 12.9 s); the times in `tabla_L.csv` come from a fully loaded node and are not the reported ones |
| `paper_L_nodos_por_instancia.csv`, `tabla_L_nodos.csv` | Test split: the port of DLTS (DFS and LDS) with the node budgets published for the original search (rows `factor_nodos = 1.0`). Column Port of the same table |
| `paper_M_g123_t100_por_instancia.csv`, `tabla_M.csv` | Validation split: control of the port against the published figures |
| `calibracion.json` | Calibration of the threshold of the port on our validation instances |

The gap reported in the paper is total moves over total reference moves minus one, i.e.
`movs / opt_medio − 1` with the columns of `tabla_L.csv`; the column `gap_pct` of these files is the
mean of the per-instance gaps and is not the reported figure.

Settings of the port in these runs, the ones frozen by the calibration: logarithmic threshold with
`p = 0.35` for DFS and `p = 0.50` for LDS, temperature 1, no bounding, and the step limit as the
initial upper bound. Folders `5x5`, `7x5` and `10x5` of the result files are `5-5`, `5-7` and `5-10`
of `instances/benchmarks/g1`–`g3` (maximum height 7):

```python
from solvers import DLTSSolver
from solvers.dlts.counters import NodeBudget

solver = DLTSSolver(action_model, input_adapter, cost_model=cost_model,
                    strategy="lds", p=0.50, bstrategy="log", temperature=1.0,
                    use_bounding=False, ub_mode="max_steps",
                    budget=NodeBudget(max_expansions=8192))
solved, steps, t = solver.solve("benchmarks/g1/5-5/0.dat", 7, 100)
```

`notebooks/dlts.ipynb` shows how to load the models and the input adapter.

## `cost_network/` — accuracy of the cost network

Analysis of an instrumented run of BS+Cost and BSG on the 760 CV instances (widths 2 to 32 with the
other evaluator as a shadow, and BS+Cost up to 512 without instrumentation). `informe.txt` is the
full report; `tab_cost_rank.tex` and `fig_cost_rank.tex` are the table and figure of the paper;
the CSV files hold the data behind each statement (`delta_por_grupo.csv`, `consultas.csv`,
`orden_decision.csv`, `orden_por_tramo.csv`, `bscost_vs_bsg.csv`, `camino_*.csv`, `raiz.csv`,
`traza_6-10_data6-10-10_w32.csv`). `meta_21611.json` records the code version and environment.

## `statistical_tests/` — paired comparisons

`wilcoxon.csv` and `wilcoxon_por_grupo.csv`: paired two-sided Wilcoxon signed-rank tests between
methods on the CV instances (BS+Cost against BS-FRG and BSG at comparable time, Greedy against
FRG). `verificacion.txt` checks the inputs of the tests against the tables of the paper.
