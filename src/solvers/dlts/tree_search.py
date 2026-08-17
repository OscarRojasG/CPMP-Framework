"""
Port del código original de DLTS (Hottung et al., 2020), DLTS-master/tree_search.py
(search_dfs, search_lds) sobre
la interfaz `solution` (aquí LayoutSolution). Se conserva la lógica original
línea a línea salvo las desviaciones D1-D6, documentadas en cada punto donde
aparecen y resumidas aquí (D6 vive en dlts_solver.py, no aquí):

  D1 - budget.exhausted(...) sustituye al timeout como condición de parada
       PRINCIPAL (decisión de presupuesto de nodos igualado); el timeout
       queda como red de seguridad opcional dentro del propio NodeBudget.
  D2 - en DFS, si get_branch_network_prediction()+get_illegal_moves() no deja
       ningún movimiento con valor > -1 (dead end), se trata como una poda
       normal (backtracking inmediato) en vez de aplicar np.argmax sobre un
       vector todo -1 (bug del original: aplicaba un movimiento ilegal que
       corrompe el estado en silencio).
  D3 - start_incumbent permite un upper bound inicial "con solución" (warm
       start): sin esto, si la búsqueda no mejora start_ub, devolvería
       incumbent=[] (== "no resuelta"), lo que sesgaría la ablación de UB.
  D4 - np.errstate(divide='ignore') alrededor de mp_log (branch_functions.py);
       cosmético, mismos valores (-inf) que el original.
  D5 - se registra un punto (nodes_expanded, ub) en checkpoints configurables
       de NodeBudget.checkpoints. Como el presupuesto sólo corta el bucle sin
       alterar ninguna decisión de búsqueda, la trayectoria con presupuesto B
       es un prefijo exacto de la de 8B: una sola corrida al presupuesto
       máximo entrega la curva anytime completa.
  D6 - (en DLTSSolver.solve_from_layout, no aquí) un layout que YA está
       ordenado en la raíz se resuelve con 0 movimientos sin entrar a buscar.
       search_dfs/search_lds sólo declaran "resuelto" con incumbent no vacío,
       así que sin esto el óptimo trivial (0 movs) se reporta como NO resuelta
       -- caso real del benchmark: benchmarks/3-3/data3-3-39.dat.

Todo lo demás se replica sin tocar: la pila DFS con el vector completo de
probabilidades y los ya tomados a -1, el undo antes de mirar stack[-1], la
profundidad POSTERIOR al undo dentro de branch_func, la discrepancia = rango
del hijo en LDS, el binning, zero_depth, y las dos podas (cota trivial y
bounding cada n niveles).
"""
import copy
import math
import time
from dataclasses import dataclass
from heapq import heappush, heappop

import numpy as np

from solvers.dlts.branch_functions import get_branch_func
from solvers.dlts.counters import NodeBudget, SearchStats


class DFSTreeNode:
    __slots__ = ("max_prob", "branches")

    def __init__(self, max_prob, branches):
        self.max_prob = max_prob
        self.branches = branches


class LDSTreeNode:
    __slots__ = ("discrepency", "depth", "pre_solution", "branch_num")

    def __init__(self, discrepency, depth, pre_solution, branch_num):
        self.discrepency = discrepency
        self.depth = depth
        self.pre_solution = pre_solution
        self.branch_num = branch_num

    def __lt__(self, other):
        if self.discrepency == other.discrepency:
            return self.depth > other.depth
        return self.discrepency < other.discrepency


@dataclass
class SearchResult:
    incumbent: list
    ub: float
    stats: SearchStats


class _CheckpointTracker:
    """Registra puntos de la curva anytime al cruzar umbrales de NODOS y de
    SEGUNDOS. Ambas rejillas son monótonas, así que basta un índice por rejilla
    y el coste por iteración es una comparación (más una lectura del reloj sólo
    mientras queden cortes de tiempo pendientes)."""

    def __init__(self, stats: SearchStats, checkpoints, time_checkpoints=()):
        self.stats = stats
        self.checkpoints = sorted(checkpoints)
        self.time_checkpoints = sorted(time_checkpoints)
        self.idx = 0
        self.t_idx = 0

    def maybe_record(self, ub) -> None:
        t = None
        if self.t_idx < len(self.time_checkpoints):
            t = self.stats.elapsed()
        while self.idx < len(self.checkpoints) and self.stats.nodes_expanded >= self.checkpoints[self.idx]:
            self.stats.record_checkpoint(ub, t)
            self.idx += 1
        while self.t_idx < len(self.time_checkpoints) and t >= self.time_checkpoints[self.t_idx]:
            self.stats.record_checkpoint(ub, t)
            self.t_idx += 1


def _backtrack(stack, solution, ub, pp, branch_func):
    """Deshace nodos hasta encontrar un hermano no probado que supere el
    umbral MP, o hasta vaciar la pila. Devuelve True si se aplicó un
    movimiento (la búsqueda sigue), False si la pila quedó vacía."""
    while stack:
        solution.undo_last_move()
        tn = stack[-1]
        branch_values = tn.branches
        max_prob = tn.max_prob
        move = np.argmax(branch_values)
        if branch_values[move] > -1 and \
                branch_values[move] >= branch_func(max_prob, ub, solution.get_cost(), pp):
            branch_values[move] = -1
            solution.apply(move)
            return True
        stack.pop()
    return False


def search_dfs(solution, use_lb_network, start_ub, pp, dd=0.95, nn=3, bstrategy='log',
               budget: NodeBudget | None = None,
               start_incumbent: list | None = None,
               stats: SearchStats | None = None) -> SearchResult:
    start_time = time.process_time()
    stack = []
    ub = start_ub
    incumbent = list(start_incumbent) if start_incumbent else []
    root = True

    stats = stats if stats is not None else SearchStats()
    stats.start_clock(start_time)
    budget = budget if budget is not None else NodeBudget()
    ckpt = _CheckpointTracker(stats, budget.checkpoints, budget.time_checkpoints)
    branch_func = get_branch_func(bstrategy)

    while stack or root:
        reason = budget.exhausted(stats, start_time)
        if reason is not None:
            stats.stop_reason = reason
            break

        root = False
        stats.nodes_visited += 1
        stats.legacy_node_count += 1
        depth = solution.get_cost()
        stats.record_depth(depth)
        stats.record_open_list(len(stack))

        if solution.is_complete():
            cost = solution.get_cost()
            if ub > cost:
                ub = cost
                incumbent = solution.get_move_list().copy()
                stats.incumbent_updates += 1

        depth = solution.get_cost()
        poda = depth + 1 >= ub or (
            use_lb_network and depth % nn == 0 and
            depth + (solution.get_lb_network_prediction() * dd) + 1 > ub
        )

        if poda:
            _backtrack(stack, solution, ub, pp, branch_func)
        else:
            branch_values = solution.get_branch_network_prediction()
            illegal_moves = solution.get_illegal_moves()
            branch_values[illegal_moves] = -1
            stats.nodes_expanded += 1
            ckpt.maybe_record(ub)

            if not np.any(branch_values > -1):
                # D2: dead end -- se trata como una poda (backtracking
                # inmediato), no se aplica ningún movimiento.
                stats.dead_ends += 1
                _backtrack(stack, solution, ub, pp, branch_func)
            else:
                move = np.argmax(branch_values)
                max_prob = branch_values[move]
                threshold = branch_func(max_prob, ub, depth, pp)
                stats.nodes_generated += int(np.sum(branch_values >= threshold))

                tn = DFSTreeNode(max_prob, branch_values)
                stack.append(tn)
                branch_values[move] = -1
                solution.apply(move)

    stats.time_s = time.process_time() - start_time
    return SearchResult(incumbent=incumbent, ub=ub, stats=stats)


def push_children_lds(heap, node, solution, branch_func, ub, pp, use_bins, nbins, zero_depth):
    """Post: todos los hijos factibles que no exceden la cota MP se empujan
    al heap. Devuelve cuántos se empujaron (para stats; el original no lo
    reporta)."""
    branch_values = solution.get_branch_network_prediction()
    illegal_moves = solution.get_illegal_moves()
    branch_values[illegal_moves] = -1
    max_prob = max(branch_values)
    sort_mp = [(ii, vv) for ii, vv in enumerate(branch_values) if vv > -1]
    sort_mp.sort(key=lambda xx: xx[1], reverse=True)
    new_depth = node.depth + 1
    if use_bins:
        bin_size = max_prob / nbins

    n_pushed = 0
    for disc, (move_num, mp) in enumerate(sort_mp):
        if mp >= branch_func(max_prob, ub, solution.get_cost(), pp):
            new_discrepency = disc
            if use_bins:
                new_discrepency = nbins - int(math.ceil(mp / bin_size))
            if new_depth <= zero_depth:
                new_discrepency = 0
            tn = LDSTreeNode(node.discrepency + new_discrepency, new_depth, solution, move_num)
            heappush(heap, tn)
            n_pushed += 1
    return n_pushed


def search_lds(start_solution, use_lb_network, start_ub, pp, dd=0.95, nn=3, bstrategy='log',
               use_bins=False, nbins=5, zero_depth=-1,
               budget: NodeBudget | None = None,
               start_incumbent: list | None = None,
               stats: SearchStats | None = None) -> SearchResult:
    start_time = time.process_time()
    incumbent = list(start_incumbent) if start_incumbent else []
    ub = start_ub

    stats = stats if stats is not None else SearchStats()
    stats.start_clock(start_time)
    budget = budget if budget is not None else NodeBudget()
    ckpt = _CheckpointTracker(stats, budget.checkpoints, budget.time_checkpoints)
    branch_func = get_branch_func(bstrategy)

    heap = []
    root = LDSTreeNode(0, 0, start_solution, 0)
    heappush(heap, root)

    while heap:
        reason = budget.exhausted(stats, start_time)
        if reason is not None:
            stats.stop_reason = reason
            break

        stats.record_open_list(len(heap))
        cur = heappop(heap)
        solution = copy.deepcopy(cur.pre_solution)
        stats.nodes_visited += 1
        stats.legacy_node_count += 1
        stats.record_depth(cur.depth)
        if cur.depth != 0:
            solution.apply(cur.branch_num)

        if solution.is_complete() and cur.depth < ub:
            ub = cur.depth
            incumbent = solution.get_move_list().copy()
            stats.incumbent_updates += 1
        elif cur.depth + 1 < ub:
            expand = (
                not use_lb_network or cur.depth % nn != 0 or
                cur.depth + 1 + (solution.get_lb_network_prediction() * dd) < ub
            )
            if expand:
                stats.nodes_expanded += 1
                ckpt.maybe_record(ub)
                n_pushed = push_children_lds(heap, cur, solution, branch_func, ub, pp, use_bins, nbins, zero_depth)
                stats.nodes_generated += n_pushed

    stats.time_s = time.process_time() - start_time
    stats.record_open_list(len(heap))
    return SearchResult(incumbent=incumbent, ub=ub, stats=stats)
