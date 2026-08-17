"""
Puente entre Layout y las redes: política (branching) y valor (bounding).

Decisiones de diseño (ver plan/README):
- softmax sobre los 90 logits del action model, temperatura configurable. Es
  numéricamente equivalente a un softmax restringido a las acciones legales
  porque exp(-1e4 - max_logit) subdesborda a 0.0 en float32/float64 (los
  logits son q.k^T/sqrt(64), magnitud muy menor a 1e4). No hace falta
  reindexar a S*(S-1): el soporte de la distribución ya cae exactamente en
  las acciones con src,dst < S.
- Guard de dead end: si el máximo logit crudo es <= -1e3, no queda ninguna
  acción legal según la red -> policy() devuelve None.
- Nunca se llama a model(...) (heredado de Transformer.forward, que descarta la
  caché de embeddings al terminar el forward). Se llama a encode(..., memory=...)
  + decode(...) explícitamente, pasando un diccionario de caché propio del
  NetProvider (self._encode_memory) que sobrevive a toda la búsqueda: la caché de
  CPMPTransformer.encode es POR PILA (clave = L y X aplanados de esa pila, que
  determinan el embedding por completo; la máscara s_mask se aplica después de
  leerla), y los estados hermanos de un árbol DFS/LDS comparten casi todas sus
  pilas. Es una caché estrictamente más fina que el memo por estado de abajo.
- value() = exp(cost_model(...)) porque el cost model se entrena para
  predecir log(pasos restantes) (ver generation/adapters/output/cost_adapter.py).
  Es V^pi (bajo nuestra política), no V*: d*V_hat NO es una cota inferior
  válida en sentido estricto y puede podar el óptimo; se mitiga calibrando
  `d` empíricamente (ver DLTS/calibrate.py), igual que hace el paper con su
  propio d=0.95 heredado.
- Memoización opcional (LRU simple) por instancia: el presupuesto de nodos se
  contabiliza sobre las consultas LÓGICAS (policy_calls/value_calls), no
  sobre los forwards reales, para que la comparación no dependa de cuánta
  caché tenga cada método.
"""
from collections import OrderedDict

import numpy as np
import torch

from solvers.dlts.layout_ops import state_key
from solvers.dlts.counters import SearchStats


# ---------------------------------------------------------------------------
# Codificación de acciones en el espacio fijo de 90 = S_max*(S_max-1) índices.
# Idéntica a la de action_adapter.py / ModelSolver / BSGCostPredictorSolver.
# ---------------------------------------------------------------------------

def idx90_to_move(idx: int, s_max: int = 10) -> tuple[int, int]:
    src = idx // (s_max - 1)
    r = idx % (s_max - 1)
    dst = r if r < src else r + 1
    return src, dst


def move_to_idx90(src: int, dst: int, s_max: int = 10) -> int:
    r = dst if dst < src else dst - 1
    return src * (s_max - 1) + r


def idx90_to_idxS(idx: int, S: int, s_max: int = 10) -> int | None:
    """Convierte un índice del espacio de 90 al espacio S*(S-1) que usa DLTS
    (útil sólo para tests cruzados y volcados legibles). None si cae fuera de
    las primeras S pilas."""
    src, dst = idx90_to_move(idx, s_max)
    if src >= S or dst >= S:
        return None
    r = dst if dst < src else dst - 1
    return src * (S - 1) + r


def idxS_to_idx90(idx: int, S: int, s_max: int = 10) -> int:
    src = idx // (S - 1)
    r = idx % (S - 1)
    dst = r if r < src else r + 1
    return move_to_idx90(src, dst, s_max)


# ---------------------------------------------------------------------------
# NetProvider
# ---------------------------------------------------------------------------

DEAD_END_THRESHOLD = -1e3  # los logits enmascarados por decode() valen -1e4


class NetProvider:
    def __init__(self, action_model, cost_model, input_adapter, H,
                 temperature: float = 1.0,
                 memo: bool = True, memo_max: int = 200_000,
                 stats: SearchStats | None = None,
                 device: str = "cpu"):
        self.action_model = action_model
        self.cost_model = cost_model
        self.input_adapter = input_adapter
        self.H = H
        self.temperature = temperature
        self.memo = memo
        self.memo_max = memo_max
        self.stats = stats if stats is not None else SearchStats()
        self.device = device

        self._policy_memo: "OrderedDict" = OrderedDict()
        self._value_memo: "OrderedDict" = OrderedDict()

        # Caché por PILA de CPMPTransformer.encode, compartida por toda la
        # búsqueda (ver docstring del módulo). No es un OrderedDict porque el
        # modelo la escribe directamente; se vacía entera al pasarse de tamaño.
        self._encode_memory: dict = {}

    def _to_batch(self, L, X, S, H):
        tensors = []
        for val in (L, X, S, H):
            if isinstance(val, np.ndarray):
                t = torch.from_numpy(val).unsqueeze(0)
            else:
                t = torch.tensor([val])
            tensors.append(t.to(self.device))
        return tensors

    def policy(self, layout) -> np.ndarray | None:
        """Distribución de probabilidad [90] sobre movimientos, o None si no
        queda ninguna acción legal según el enmascarado de la propia red."""
        key = state_key(layout)
        if self.memo and key in self._policy_memo:
            self._policy_memo.move_to_end(key)
            self.stats.policy_calls += 1
            return self._policy_memo[key]

        L, X, S, H = self.input_adapter.input_2_vec(layout, self.H)
        tensors = self._to_batch(L, X, S, H)

        if len(self._encode_memory) > self.memo_max:
            self._encode_memory = {}

        with torch.no_grad():
            stack_embeddings, self._encode_memory = self.action_model.encode(
                *tensors, memory=self._encode_memory)
            logits = self.action_model.decode(stack_embeddings, *tensors)
        logits = logits[0]

        self.stats.policy_calls += 1
        self.stats.policy_forwards += 1

        if logits.max().item() <= DEAD_END_THRESHOLD:
            result = None
        else:
            with torch.no_grad():
                probs = torch.softmax(logits / self.temperature, dim=-1)
            result = probs.cpu().numpy().astype(np.float64)

        if self.memo:
            self._policy_memo[key] = result
            if len(self._policy_memo) > self.memo_max:
                self._policy_memo.popitem(last=False)

        return result

    def value(self, layout) -> float:
        """exp(cost_model(...)) = estimación de pasos restantes (V^pi)."""
        if self.cost_model is None:
            raise RuntimeError("NetProvider sin cost_model: no se puede consultar el bounding")

        key = state_key(layout)
        if self.memo and key in self._value_memo:
            self._value_memo.move_to_end(key)
            self.stats.value_calls += 1
            return self._value_memo[key]

        L, X, S, H = self.input_adapter.input_2_vec(layout, self.H)
        tensors = self._to_batch(L, X, S, H)

        with torch.no_grad():
            stack_embeddings, _ = self.cost_model.encode(*tensors, memory=None)
            pred = self.cost_model.decode(stack_embeddings, *tensors)
        log_cost = pred[0].item()
        result = float(np.exp(log_cost))

        self.stats.value_calls += 1
        self.stats.value_forwards += 1

        if self.memo:
            self._value_memo[key] = result
            if len(self._value_memo) > self.memo_max:
                self._value_memo.popitem(last=False)

        return result

    def clear(self) -> None:
        self._policy_memo.clear()
        self._value_memo.clear()
        self._encode_memory = {}
