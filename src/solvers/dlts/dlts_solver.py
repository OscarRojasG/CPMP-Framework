"""
DLTSSolver: envuelve tree_search.search_dfs/search_lds con la interfaz Solver
del proyecto ((solved, steps, time), compatible con Solver.solve_from_folder).
"""
import time
from dataclasses import replace

from solvers.solver import Solver
from solvers.dlts.counters import NodeBudget, SearchStats
from solvers.dlts.layout_ops import clone_layout
from solvers.dlts.nets import NetProvider, idx90_to_move, move_to_idx90
from solvers.dlts.search_node import LayoutSolution
from solvers.dlts.tree_search import search_dfs, search_lds


class DLTSSolver(Solver):
    def __init__(self, action_model, input_adapter, cost_model=None, *,
                 strategy: str = "dfs",              # 'dfs' | 'lds'
                 p: float = 0.1, d: float = 0.95, n: int = 3,
                 bstrategy: str = "log",
                 use_bins: bool = False, nbins: int = 5, zero_depth: int = -1,
                 use_bounding: bool = False,          # ablación (b)
                 ub_mode: str = "max_steps",           # 'max_steps' | 'greedy' -- ablación (c)
                 budget: NodeBudget | None = None,
                 temperature: float = 1.0,
                 forbid_undo: str = "dlts", cycle_mode: str = "off",
                 memo: bool = True, s_max: int | None = None):
        super().__init__(f"DLTS-{strategy.upper()}")

        if strategy not in ("dfs", "lds"):
            raise ValueError(f"strategy debe ser 'dfs' o 'lds', no {strategy!r}")
        if ub_mode not in ("max_steps", "greedy"):
            raise ValueError(f"ub_mode debe ser 'max_steps' o 'greedy', no {ub_mode!r}")
        if use_bounding and cost_model is None:
            raise ValueError("use_bounding=True requiere pasar cost_model")

        self.action_model = action_model
        self.cost_model = cost_model
        self.input_adapter = input_adapter
        self.strategy = strategy
        self.p = p
        self.d = d
        self.n = n
        self.bstrategy = bstrategy
        self.use_bins = use_bins
        self.nbins = nbins
        self.zero_depth = zero_depth
        self.use_bounding = use_bounding
        self.ub_mode = ub_mode
        self.budget = budget
        self.temperature = temperature
        self.forbid_undo = forbid_undo
        self.cycle_mode = cycle_mode
        self.memo = memo
        # El espacio de acciones que produce decode() es S_max*(S_max-1) con el
        # S_max del adaptador, y los scripts de este repo lo construyen POR
        # CARPETA (S_max = S, H_max = H). Derivarlo evita el desajuste silencioso
        # que causaría fijar 10 aquí: las acciones se decodificarían mal.
        self.s_max = s_max if s_max is not None else input_adapter.S_max

        self._stats_history: list = []

    def solve_from_layouts(self, layouts, H, max_steps):
        results = []
        for layout in layouts:
            solved, steps, _t = self.solve_from_layout(layout, H, max_steps)
            results.append([solved, steps])
        return results

    def solve_from_layout(self, layout, H, max_steps):
        t0 = time.perf_counter()

        layout_initial = clone_layout(layout)   # se preserva para validar (V7)
        search_layout = clone_layout(layout)    # la que mutará la búsqueda

        stats = SearchStats()

        # D6: instancia ya ordenada en la raíz -> 0 movimientos. search_dfs/
        # search_lds sólo declaran "resuelto" cuando el incumbent es no vacío
        # (len(incumbent) > 0), así que un layout que YA está ordenado termina
        # con incumbent=[] y se reporta como NO resuelta. Es un caso real del
        # benchmark (benchmarks/3-3/data3-3-39.dat). Se resuelve en el wrapper
        # y no en tree_search.py para no alterar el algoritmo portado ni la
        # equivalencia exacta con el original que verifica V4.
        if search_layout.is_sorted():
            stats.stop_reason = "already_sorted"
            stats.time_s = time.perf_counter() - t0
            self._stats_history.append(stats)
            return True, 0, stats.time_s

        nets = NetProvider(
            self.action_model,
            self.cost_model if self.use_bounding else None,
            self.input_adapter, H,
            temperature=self.temperature, memo=self.memo, stats=stats,
        )

        warmstart_nodes = 0
        start_incumbent = None
        if self.ub_mode == "greedy":
            from solvers.model import ModelSolver  # import perezoso: evita ciclo con solvers/__init__

            # ModelSolver muta el layout que recibe, así que greedy_layout.moves
            # queda con el rollout completo y sirve de incumbent inicial.
            greedy_layout = clone_layout(layout)
            greedy_solver = ModelSolver(self.action_model, self.input_adapter)
            g_solved, g_steps, _gt = greedy_solver.solve_from_layout(greedy_layout, H, max_steps)
            # En greedy, un nodo expandido = una consulta a la política = un paso
            # aplicado, así que steps es el conteo exacto de nodos del warm start
            # (lo que necesita la regla R7 más abajo).
            warmstart_nodes = greedy_layout.steps

            if g_solved:
                start_ub = g_steps
                start_incumbent = [move_to_idx90(src, dst, self.s_max) for src, dst in greedy_layout.moves]
            else:
                start_ub = max_steps
        else:
            start_ub = max_steps

        stats.warmstart_nodes = warmstart_nodes

        budget = self.budget if self.budget is not None else NodeBudget()
        if warmstart_nodes and budget.max_expansions is not None:
            # R7: los nodos del warm start se descuentan del mismo presupuesto
            budget = replace(budget, max_expansions=max(0, budget.max_expansions - warmstart_nodes))

        sol = LayoutSolution(
            search_layout, H, nets, s_max=self.s_max,
            forbid_undo=self.forbid_undo, cycle_mode=self.cycle_mode,
        )

        if self.strategy == "dfs":
            result = search_dfs(
                sol, self.use_bounding, start_ub, self.p, dd=self.d, nn=self.n,
                bstrategy=self.bstrategy, budget=budget,
                start_incumbent=start_incumbent, stats=stats,
            )
        else:
            result = search_lds(
                sol, self.use_bounding, start_ub, self.p, dd=self.d, nn=self.n,
                bstrategy=self.bstrategy, use_bins=self.use_bins, nbins=self.nbins,
                zero_depth=self.zero_depth, budget=budget,
                start_incumbent=start_incumbent, stats=stats,
            )

        solved = len(result.incumbent) > 0
        steps = len(result.incumbent) if solved else float("inf")

        if solved:
            self._validate_solution(layout_initial, result.incumbent, steps)

        t1 = time.perf_counter()
        stats.time_s = t1 - t0
        self._stats_history.append(stats)

        return solved, steps, stats.time_s

    def _validate_solution(self, layout_initial, incumbent, expected_steps) -> None:
        """V7: re-aplica el incumbent desde el layout inicial y verifica que
        deja el bay ordenado. Cuesta microsegundos; cualquier fallo aquí es
        un bug de undo/deepcopy, no un problema de calidad de la búsqueda."""
        check = clone_layout(layout_initial)
        for idx in incumbent:
            src, dst = idx90_to_move(idx, self.s_max)
            check.move(src, dst)
        if not check.is_sorted():
            raise RuntimeError(
                f"{self.name}: solución inválida -- re-aplicar el incumbent no deja el layout ordenado"
            )
        if len(check.moves) != expected_steps:
            raise RuntimeError(f"{self.name}: longitud de incumbent inconsistente con steps reportado")

    @property
    def last_stats(self) -> list:
        return [s.as_dict() for s in self._stats_history]

    def reset(self) -> None:
        self._stats_history = []
