import torch
from solvers.solver import Solver
import copy
import time


class ModelSolver(Solver): 
    def __init__(self, model, input_adapter, batch_size=32):
        super().__init__("ModelSolver")
        self.model = model
        self.input_adapter = input_adapter
        self.batch_size = batch_size

        # Diccionario de embeddings
        self.memory = None

    def solve_from_layouts(self, layouts, H, max_steps):
        results = []
        for i in range(0, len(layouts), self.batch_size):
            batch_layouts = layouts[i:i+self.batch_size]
            r = self.solve_batch(batch_layouts, H, max_steps)
            results += r

        return results
    
    def solve_from_layout(self, layout, H, max_steps):
        t0 = time.perf_counter()
        result = self.solve_from_layouts([layout], H, max_steps)[0]
        t1 = time.perf_counter()
        t = t1 - t0

        return *result, t

    def solve_batch(self, layouts, H, max_steps):
        S = len(layouts[0].stacks)
        num_layouts = len(layouts)
        
        # Historial de estados visitados por cada layout individualmente
        visited_states_list = [set() for _ in range(num_layouts)]
        
        # Conjunto de layouts descartados por acción inválida (quedan como "no resuelto")
        discarded_indices = set()

        with torch.no_grad():
            while any(
                not l.is_sorted() and l.steps < max_steps and i not in discarded_indices
                for i, l in enumerate(layouts)
            ):
                # Identificamos qué layouts aún necesitan procesarse
                active_indices = [
                    i for i, l in enumerate(layouts)
                    if not l.is_sorted() and l.steps < max_steps and i not in discarded_indices
                ]

                # Guardamos estados actuales de los layouts activos
                for i in active_indices:
                    current_state = tuple(tuple(stack) for stack in layouts[i].stacks)
                    visited_states_list[i].add(current_state)

                # Preparación del batch de datos
                batch_data_lists = []
                for i in active_indices:
                    data = list(self.input_adapter.input_2_vec(layouts[i], H))
                    for j in range(len(data)):
                        val = data[j]
                        data[j] = torch.tensor([val]) if isinstance(val, (int, float)) else torch.from_numpy(val).unsqueeze(0)
                    batch_data_lists.append(data)

                # Empaquetamos en tensores de batch: [batch_size, ...]
                batch_inputs = [torch.cat(tensors, dim=0) for tensors in zip(*batch_data_lists)]

                # Inferencia en batch
                stack_embeddings, self.memory = self.model.encode(*batch_inputs, memory=self.memory)
                logits = self.model.decode(stack_embeddings, *batch_inputs)

                # Ordenamos índices de mejor a peor; capturamos también los valores para validar
                top_values_batch, top_indices_batch = torch.sort(logits, dim=1, descending=True)

                # Aplicamos la lógica de movimiento individualmente
                for idx_in_batch, original_idx in enumerate(active_indices):
                    layout = layouts[original_idx]
                    top_indices = top_indices_batch[idx_in_batch]
                    top_values  = top_values_batch[idx_in_batch]
                    visited_states = visited_states_list[original_idx]

                    for i in range(len(top_indices)):
                        best_index  = top_indices[i].item()
                        best_value  = top_values[i].item()

                        # --- Acción inválida enmascarada: descartar layout inmediatamente ---
                        if best_value <= -1e4:
                            discarded_indices.add(original_idx)
                            break
                        # -----------------------------------------------------------------

                        src = int(best_index / (self.input_adapter.S_max - 1))
                        r   = best_index % (self.input_adapter.S_max - 1)
                        dst = r if r < src else r + 1

                        # Previsualización del movimiento
                        temp_layout = copy.deepcopy(layout)
                        temp_layout.move(src, dst)
                        next_state = tuple(tuple(stack) for stack in temp_layout.stacks)

                        if next_state not in visited_states:
                            layout.move(src, dst)
                            break

        # Resultados finales
        # Los layouts descartados cuentan como "no resuelto" (False)
        results = [(l.unsorted_stacks == 0, l.steps) for l in layouts]
        return results
    
    def reset(self):
        self.memory = None