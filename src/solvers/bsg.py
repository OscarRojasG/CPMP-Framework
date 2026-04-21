import torch
from solvers.solver import Solver
from solvers.model import ModelSolver
import copy
from generation.adapters import *


class BSGSolver(Solver): 
    def __init__(self, model, layout_adapter, w, batch_size=32):
        super().__init__("ModelSolver")
        self.model = model
        self.layout_adapter = layout_adapter
        self.w = w
        self.batch_size = batch_size

    def solve_from_layouts(self, layouts, H, max_steps):
        results = []
        for layout in layouts:
            r = self.solve_from_layout(layout, H, max_steps)
            results.append(r)

        return results
    
    def solve_from_layout(self, layout, H, max_steps):
        states = []
        states.append(layout)
        best_state = None
        model_solver = ModelSolver(self.model, self.layout_adapter, self.batch_size)
        visited_states = set()

        while not best_state and states[0].steps < max_steps:
            children = []
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i+self.batch_size]
                children += self.expand(batch_states, visited_states, H)

            evals = []
            for i in range(0, len(children), self.batch_size):
                batch_children = children[i:i+self.batch_size]
                evals += self.eval(model_solver, batch_children, H, max_steps)

            evals = torch.tensor(evals)
            _, indices = torch.topk(evals, k=self.w, largest=False)
            states = [children[i] for i in indices]

            for state in states:
                if state.is_sorted():
                    best_state = state
                    break

        if best_state:
            return True, best_state.steps
        return False, float('inf')
    
    def expand(self, states, visited_states, H):
        S = len(states[0].stacks)
        children = []

        # Preparación del batch de datos
        batch_data_lists = []
        for state in states:
            data = list(self.layout_adapter.layout_2_vec(state, H))
            for j in range(len(data)):
                val = data[j]
                data[j] = torch.tensor([val]) if isinstance(val, (int, float)) else torch.from_numpy(val).unsqueeze(0)
            batch_data_lists.append(data)

        batch_inputs = [torch.cat(tensors, dim=0) for tensors in zip(*batch_data_lists)]
        
        # Inferencia en batch
        logits = self.model(*batch_inputs)
        
        # Ordenamos índices de mejor a peor para cada layout en el batch
        top_values_batch, top_indices_batch = torch.sort(logits, dim=1, descending=True)

        # Aplicamos la lógica de movimiento individualmente
        for idx, state in enumerate(states):
            top_indices = top_indices_batch[idx]
            top_values = top_values_batch[idx]
            children_count = 0

            for i in range(len(top_indices)):
                if top_values[i] < -9000:
                    break

                best_index = top_indices[i].item()
                src = int(best_index / (S - 1))
                r = best_index % (S - 1)
                dst = r if r < src else r + 1

                # Previsualización del movimiento
                child_layout = copy.deepcopy(state)
                child_layout.move(src, dst)
                next_state = tuple(tuple(stack) for stack in child_layout.stacks)
                
                if next_state not in visited_states:
                    children.append(child_layout)
                    visited_states.add(next_state)
                    children_count += 1
                    if children_count >= self.w:
                        break

        return children
    
    def eval(self, model_solver, children, H, max_steps):
        children_copy = [copy.deepcopy(child) for child in children]
        result = model_solver.solve_from_layouts(children_copy, H, max_steps)
        return [r[1] for r in result]