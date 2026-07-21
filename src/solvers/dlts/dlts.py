import time
import copy
import solvers.dlts.tree_search as tree_search
from solvers.dlts.pytorch_node import PyTorchSearchNode
from solvers.solver import Solver

class DLTSSolver(Solver):
    def __init__(self, model, input_adapter, value_model=None, 
                 search_strategy='dfs', ts_param_p=0.1, ts_param_d=0.95, 
                 ts_param_n=3, timeout=60.0, bstrategy='log'):
        super().__init__("DLTSSolver")
        self.model = model
        self.value_model = value_model
        self.input_adapter = input_adapter
        
        # Parámetros del Tree Search original
        self.search_strategy = search_strategy.lower()
        self.ts_param_p = ts_param_p
        self.ts_param_d = ts_param_d
        self.ts_param_n = ts_param_n
        self.timeout = timeout
        self.bstrategy = bstrategy

    def solve_from_layout(self, layout, H, max_steps):
        """
        Evalúa una instancia a la vez.
        Retorna: solved (bool), steps (int), time (float)
        """
        # Clonamos el layout profundo para que la búsqueda en árbol no mute 
        # tu objeto original permanentemente (ya que el árbol explora y deshace movimientos)
        layout_copy = copy.deepcopy(layout)
        
        visited_memory = {}
        
        node = PyTorchSearchNode(
            layout=layout_copy, 
            H=H, 
            input_adapter=self.input_adapter, 
            branch_model=self.model, 
            value_model=self.value_model,
            visited_memory=visited_memory
        )
        
        has_value_model = self.value_model is not None
        
        # Usamos perf_counter para precisión en el tiempo de ejecución (igual que tu ModelSolver)
        t0 = time.perf_counter()
        
        # Ejecutamos la estrategia de búsqueda seleccionada
        if self.search_strategy == 'dfs':
            res = tree_search.search_dfs(
                node, has_value_model, max_steps, self.ts_param_p, 
                self.ts_param_d, self.ts_param_n, verbose=0, 
                timeout=self.timeout, bstrategy=self.bstrategy
            )
        elif self.search_strategy == 'lds':
            # LDS requiere algunos parámetros extra, usaremos los defaults de 2016
            res = tree_search.search_lds(
                node, has_value_model, max_steps, self.ts_param_p, 
                self.ts_param_d, self.ts_param_n, verbose=0, 
                timeout=self.timeout, bstrategy=self.bstrategy, 
                use_bins=False, nbins=5, zero_depth=-1
            )
        elif self.search_strategy == 'wbs':
            res = tree_search.search_wbs(
                node, has_value_model, max_steps, self.ts_param_p, 
                self.ts_param_d, 1.0, self.ts_param_n, verbose=0, 
                timeout=self.timeout, bstrategy=self.bstrategy, ts_param_e=0.95
            )
        else:
            raise ValueError(f"Estrategia de búsqueda desconocida: {self.search_strategy}")
            
        # Desempaquetamos el resultado del árbol original
        move_list, nodes_count, _ = res
        t1 = time.perf_counter()
        total_time = t1 - t0
        
        steps = len(move_list)
        
        # --- AGREGAR ESTA LÍNEA PARA DIAGNÓSTICO ---
        print(f"Instancia resuelta en {total_time:.2f}s | Movimientos: {steps} | NODOS EXPLORADOS: {nodes_count}")
        
        solved = steps > 0 or layout.is_sorted()
        
        return solved, steps, total_time