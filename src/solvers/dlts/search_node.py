"""
LayoutSolution: adapta cpmp.layout.Layout a la interfaz `solution` que exigen
search_dfs/search_lds (ver DLTS-master/solution_cpmp.py en el código original):
apply, undo_last_move, is_complete, get_cost, get_move_list,
get_illegal_moves, get_branch_network_prediction, get_lb_network_prediction,
__deepcopy__.

Puntos críticos:
- __deepcopy__ es imprescindible: sin él, copy.deepcopy en cada heappop de
  LDS intentaría copiar el NetProvider (y por tanto el modelo) entero. Copia
  sólo el Layout (vía clone_layout); nets se comparte por referencia a
  propósito (misma memoización).
- get_illegal_moves() replica la regla extra de DLTS ("no mover nada desde la
  pila a la que se acaba de mover", forbid_undo='dlts', estrictamente más
  fuerte que "no deshacer el último movimiento") y NO implementa detección de
  ciclos por defecto (cycle_mode='off'), fiel al paper, que dice
  explícitamente que no la implementan.
- La detección de "dead end" (ningún movimiento legal) vive en el buscador,
  nunca falsificando get_cost() (ver el port con bugs en
  CPMP-Framework-Oscar/src/solvers/dlts/pytorch_node.py, que corrompe `dd`
  dentro de branch_func haciendo eso).
"""
import numpy as np

from solvers.dlts.layout_ops import (
    apply_move,
    clone_layout,
    state_key,
    undo_last_move as _undo_last_move_layout,
)
from solvers.dlts.nets import idx90_to_move, move_to_idx90


class LayoutSolution:
    def __init__(self, layout, H, nets, s_max: int = 10,
                 forbid_undo: str = "dlts",   # 'dlts' | 'exact' | 'off'
                 cycle_mode: str = "off",     # 'off' | 'path' | 'global'
                 global_visited: set | None = None):
        self.layout = layout
        self.H = H
        self.nets = nets
        self.s_max = s_max
        self.forbid_undo = forbid_undo
        self.cycle_mode = cycle_mode

        self._global_visited = global_visited if global_visited is not None else set()
        if cycle_mode == "path":
            self._path_states = {state_key(layout)}
            self._path_stack = []
        else:
            self._path_states = None
            self._path_stack = None
        if cycle_mode == "global":
            self._global_visited.add(state_key(layout))

    # --- interfaz DLTS ----------------------------------------------------

    def apply(self, move: int) -> None:
        src, dst = idx90_to_move(move, self.s_max)
        apply_move(self.layout, src, dst)
        if self.cycle_mode in ("path", "global"):
            key = state_key(self.layout)
            if self.cycle_mode == "path":
                self._path_states.add(key)
                self._path_stack.append(key)
            else:
                self._global_visited.add(key)

    def undo_last_move(self) -> None:
        _undo_last_move_layout(self.layout)
        if self.cycle_mode == "path":
            key = self._path_stack.pop()
            if key not in self._path_stack:
                self._path_states.discard(key)

    def is_complete(self) -> bool:
        return self.layout.is_sorted()

    def get_cost(self) -> int:
        return len(self.layout.moves)

    def get_move_list(self) -> list[int]:
        return [move_to_idx90(src, dst, self.s_max) for src, dst in self.layout.moves]

    def get_illegal_moves(self) -> np.ndarray:
        S = len(self.layout.stacks)
        H = self.layout.H
        illegal = []

        last_src = last_dst = None
        if self.layout.moves:
            last_src, last_dst = self.layout.moves[-1]

        n_actions = self.s_max * (self.s_max - 1)
        for idx in range(n_actions):
            src, dst = idx90_to_move(idx, self.s_max)

            if src >= S or dst >= S:
                illegal.append(idx)
                continue
            if len(self.layout.stacks[dst]) == H:
                illegal.append(idx)
                continue
            if len(self.layout.stacks[src]) == 0:
                illegal.append(idx)
                continue
            if self.forbid_undo == "dlts" and last_dst is not None and src == last_dst:
                illegal.append(idx)
                continue
            if self.forbid_undo == "exact" and last_dst is not None and src == last_dst and dst == last_src:
                illegal.append(idx)
                continue

            if self.cycle_mode in ("path", "global"):
                c = self.layout.stacks[src][-1]
                resulting = tuple(
                    (tuple(s[:-1]) if i == src else (tuple(s) + (c,) if i == dst else tuple(s)))
                    for i, s in enumerate(self.layout.stacks)
                )
                if self.cycle_mode == "path" and resulting in self._path_states:
                    illegal.append(idx)
                    continue
                if self.cycle_mode == "global" and resulting in self._global_visited:
                    illegal.append(idx)
                    continue

        return np.array(illegal, dtype=np.int64)

    def get_branch_network_prediction(self) -> np.ndarray:
        probs = self.nets.policy(self.layout)
        if probs is None:
            return np.zeros(self.s_max * (self.s_max - 1), dtype=np.float64)
        # copia: el buscador muta este array in-place (pone a -1 lo tomado o
        # ilegal); no debe compartir memoria con la caché de nets.
        return probs.copy()

    def get_lb_network_prediction(self) -> float:
        return self.nets.value(self.layout)

    def __deepcopy__(self, memo):
        new = LayoutSolution.__new__(LayoutSolution)
        new.layout = clone_layout(self.layout)
        new.H = self.H
        new.nets = self.nets  # NUNCA copiar: mismo modelo, misma memoización
        new.s_max = self.s_max
        new.forbid_undo = self.forbid_undo
        new.cycle_mode = self.cycle_mode
        new._global_visited = self._global_visited  # compartido a propósito
        if self.cycle_mode == "path":
            new._path_states = set(self._path_states)
            new._path_stack = list(self._path_stack)
        else:
            new._path_states = None
            new._path_stack = None
        return new
