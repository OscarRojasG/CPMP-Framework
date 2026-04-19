import torch
from solvers.solver import Solver
import copy
from generation.adapters import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from cpmp.layout import read_file


class ModelSolver(Solver): 
    def __init__(self, model, layout_adapter, batch_size=32):
        super().__init__("ModelSolver")
        self.model = model
        self.layout_adapter = layout_adapter
        self.batch_size = batch_size

    def solve_from_layouts(self, layouts, H, max_steps):
        results = []
        for i in range(0, len(layouts), self.batch_size):
            batch_layouts = layouts[i:i+self.batch_size]
            r = self.solve_batch(batch_layouts, H, max_steps)
            results += r

        return results

    def solve_batch(self, layouts, H, max_steps):
        S = len(layouts[0].stacks)
        num_layouts = len(layouts)
        
        # Historial de estados visitados por cada layout individualmente
        visited_states_list = [set() for _ in range(num_layouts)]
        
        with torch.no_grad():
            while any(not l.is_sorted() and l.steps < max_steps for l in layouts):
                # Identificamos qué layouts aún necesitan procesarse
                active_indices = [
                    i for i, l in enumerate(layouts) 
                    if not l.is_sorted() and l.steps < max_steps
                ]
                
                # Guardamos estados actuales de los layouts activos
                for i in active_indices:
                    current_state = tuple(tuple(stack) for stack in layouts[i].stacks)
                    visited_states_list[i].add(current_state)

                # Preparación del batch de datos
                batch_data_lists = []
                for i in active_indices:
                    data = list(self.layout_adapter.layout_2_vec(layouts[i], H))
                    for j in range(len(data)):
                        val = data[j]
                        data[j] = torch.tensor([val]) if isinstance(val, (int, float)) else torch.from_numpy(val).unsqueeze(0)
                    batch_data_lists.append(data)

                # Empaquetamos en tensores de batch: [batch_size, ...]
                # zip(*batch_data_lists) agrupa por tipo de entrada del modelo
                batch_inputs = [torch.cat(tensors, dim=0) for tensors in zip(*batch_data_lists)]
                
                # Inferencia en batch
                logits = self.model(*batch_inputs)
                
                # Ordenamos índices de mejor a peor para cada layout en el batch
                _, top_indices_batch = torch.sort(logits, dim=1, descending=True)

                # Aplicamos la lógica de movimiento individualmente
                for idx_in_batch, original_idx in enumerate(active_indices):
                    layout = layouts[original_idx]
                    top_indices = top_indices_batch[idx_in_batch]
                    visited_states = visited_states_list[original_idx]

                    for i in range(len(top_indices)):
                        best_index = top_indices[i].item()
                        src = int(best_index / (S - 1))
                        r = best_index % (S - 1)
                        dst = r if r < src else r + 1

                        # Previsualización del movimiento
                        temp_layout = copy.deepcopy(layout)
                        temp_layout.move(src, dst)
                        next_state = tuple(tuple(stack) for stack in temp_layout.stacks)
                        
                        if next_state not in visited_states:
                            layout.move(src, dst)
                            break

        # Resultados finales
        results = [(l.unsorted_stacks == 0, l.steps) for l in layouts]
        return results