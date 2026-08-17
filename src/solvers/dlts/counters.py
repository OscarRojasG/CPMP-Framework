"""
Métricas de nodos unificadas. El código original de DLTS es inconsistente entre
estrategias (DFS cuenta iteraciones del while, LDS cuenta heappop, WBS cuenta
inserciones); aquí se definen contadores con semántica fija y comparable entre
greedy, beam y DLTS-DFS/LDS. Ver DLTS/README.md para la tabla de instrumentación
por método.

"Nodos abiertos" es ambiguo en el paper (Tabla 4, "Avg. Opened Nodes"): ahí
significa nodos PROCESADOS (nuestro nodes_expanded), no el tamaño de la
frontera. Por eso separamos nodes_expanded de open_list_max.
"""
import time
from dataclasses import dataclass, field


@dataclass
class SearchStats:
    nodes_expanded: int = 0      # nodos donde se CONSULTA la red de política (métrica principal)
    nodes_generated: int = 0     # hijos creados tras legalidad + poda MP
    nodes_visited: int = 0       # nodos "tocados": iteración del while / pop de la frontera
    policy_calls: int = 0        # consultas lógicas a la política (== nodes_expanded en DFS/LDS)
    policy_forwards: int = 0     # forwards reales tras memoización
    value_calls: int = 0         # consultas lógicas a la red de valor (bounding)
    value_forwards: int = 0
    open_list_max: int = 0       # pico de la frontera (pila DFS / heap LDS / beam)
    depth_max: int = 0
    incumbent_updates: int = 0
    legacy_node_count: int = 0   # contador del original tal cual, para auditar contra el código de DLTS
    dead_ends: int = 0           # veces que search_dfs tomó la rama D2 (dead end sin movimientos legales)
    time_s: float = 0.0
    stop_reason: str = "exhausted"  # 'exhausted' | 'budget' | 'timeout' | 'open_list_overflow'
    warmstart_nodes: int = 0     # nodos gastados por el greedy de warm start (ub_mode='greedy')
    anytime: list = field(default_factory=list)  # [(nodes_expanded, ub, t)]
    t0: float | None = None      # process_time() al arrancar la búsqueda (lo fija tree_search)

    @property
    def nn_calls(self) -> int:
        return self.policy_calls + self.value_calls

    def start_clock(self, t0: float) -> None:
        """Ancla el reloj de la curva anytime. Se llama una vez, al inicio de
        search_dfs/search_lds, con el mismo process_time() que usa NodeBudget."""
        self.t0 = t0

    def record_open_list(self, size: int) -> None:
        if size > self.open_list_max:
            self.open_list_max = size

    def record_depth(self, depth: int) -> None:
        if depth > self.depth_max:
            self.depth_max = depth

    def elapsed(self) -> float:
        return 0.0 if self.t0 is None else time.process_time() - self.t0

    def record_checkpoint(self, ub: float, t: float | None = None) -> None:
        """Un punto de la curva anytime: (nodos, mejor coste, segundos de CPU).

        El tiempo es lo que permite releer la corrida a CUALQUIER corte de
        segundos a posteriori -- por ejemplo los 60 s del paper -- sin volver a
        ejecutar la búsqueda ni cortar por reloj (que no sería reproducible
        entre máquinas). Es process_time(), coherente con NodeBudget.exhausted
        y con el hilo único del benchmark.

        `t` se puede pasar ya calculado para no repetir la llamada al reloj
        cuando quien llama acaba de leerlo.
        """
        self.anytime.append((self.nodes_expanded, ub, self.elapsed() if t is None else t))

    def as_dict(self) -> dict:
        d = {
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "nodes_visited": self.nodes_visited,
            "policy_calls": self.policy_calls,
            "policy_forwards": self.policy_forwards,
            "value_calls": self.value_calls,
            "value_forwards": self.value_forwards,
            "nn_calls": self.nn_calls,
            "open_list_max": self.open_list_max,
            "depth_max": self.depth_max,
            "incumbent_updates": self.incumbent_updates,
            "legacy_node_count": self.legacy_node_count,
            "dead_ends": self.dead_ends,
            "time_s": self.time_s,
            "stop_reason": self.stop_reason,
            "warmstart_nodes": self.warmstart_nodes,
            "anytime": list(self.anytime),
        }
        return d


@dataclass
class NodeBudget:
    max_expansions: int | None = None      # criterio de parada PRINCIPAL
    max_nn_calls: int | None = None        # tope secundario opcional
    max_depth: int | None = None           # = max_steps del benchmark
    timeout_s: float | None = None         # red de seguridad, NO criterio principal
    max_open_nodes: int | None = None      # tope duro de memoria (mp_log no poda en la raíz)
    checkpoints: tuple = ()                # cortes anytime sobre nodes_expanded
    time_checkpoints: tuple = ()           # cortes anytime sobre SEGUNDOS de CPU

    # Por qué hacen falta los dos: la comparación EXTERNA contra el paper va en
    # nodos (su Tabla 4) y la INTERNA contra beam va en segundos. Con sólo
    # cortes de nodos la curva queda coja justo donde interesa -- en las formas
    # baratas se agota el tope de nodos mucho antes del corte de 60 s y la
    # trayectoria termina antes del punto que queremos leer.

    def exhausted(self, stats: SearchStats, start_time: float) -> str | None:
        """Devuelve el motivo de parada si el presupuesto se agotó, o None si
        la búsqueda puede continuar."""
        import time as _time

        if self.max_expansions is not None and stats.nodes_expanded >= self.max_expansions:
            return "budget"
        if self.max_nn_calls is not None and stats.nn_calls >= self.max_nn_calls:
            return "budget"
        if self.max_open_nodes is not None and stats.open_list_max >= self.max_open_nodes:
            return "open_list_overflow"
        if self.timeout_s is not None and (_time.process_time() - start_time) >= self.timeout_s:
            return "timeout"
        return None
