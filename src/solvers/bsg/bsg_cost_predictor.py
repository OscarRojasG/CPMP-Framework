import torch
from solvers.solver import Solver
import copy
import time


class BSGCostPredictorSolver(Solver): 
    def __init__(self, action_model, cost_model, input_adapter, w, batch_size=32):
        super().__init__("BSGCostPredictorSolver")
        self.action_model = action_model
        self.cost_model = cost_model
        self.input_adapter = input_adapter
        self.w = w
        self.batch_size = batch_size
    
    def solve_from_layout(self, layout, H, max_steps):
        t0 = time.perf_counter()

        states = []
        states.append(layout)
        best_state = None
        visited_states = set()
        
        # 1. Inicializamos dos memorias independientes para cada modelo
        action_memory = {}
        cost_memory = {}

        while not best_state and states[0].steps < max_steps:
            children = []
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i+self.batch_size]
                # Pasamos y actualizamos la memoria de acciones
                batch_children, action_memory = self.expand(batch_states, visited_states, H, action_memory)
                children += batch_children

            evals = []
            for i in range(0, len(children), self.batch_size):
                batch_children = children[i:i+self.batch_size]
                # Pasamos y actualizamos la memoria de costos
                batch_evals, cost_memory = self.eval(batch_children, H, cost_memory)
                evals += batch_evals

            evals = torch.tensor(evals)
            k = min(self.w, len(evals))
            _, indices = torch.topk(evals, k=k, largest=False)
            states = [children[i] for i in indices]

            for state in states:
                if state.is_sorted():
                    best_state = state
                    break

        t1 = time.perf_counter()
        t = t1 - t0

        if best_state:
            return True, best_state.steps, t
        return False, float('inf'), t
    
    def expand(self, states, visited_states, H, memory):
        children = []

        # Preparación del batch de datos
        batch_data_lists = []
        for state in states:
            data = list(self.input_adapter.input_2_vec(state, H))
            for j in range(len(data)):
                val = data[j]
                data[j] = torch.tensor([val]) if isinstance(val, (int, float)) else torch.from_numpy(val).unsqueeze(0)
            batch_data_lists.append(data)

        batch_inputs = [torch.cat(tensors, dim=0) for tensors in zip(*batch_data_lists)]
        
        # 2. Inferencia en batch usando la memoria del action_model
        with torch.no_grad():
            stack_embeddings, memory = self.action_model.encode(*batch_inputs, memory=memory)
            logits = self.action_model.decode(stack_embeddings, *batch_inputs)
        
        # Ordenamos índices de mejor a peor para cada layout en el batch
        top_values_batch, top_indices_batch = torch.sort(logits, dim=1, descending=True)

        # Aplicamos la lógica de movimiento individualmente
        for idx, state in enumerate(states):
            top_indices = top_indices_batch[idx]
            top_values = top_values_batch[idx]
            children_count = 0

            for i in range(len(top_indices)):
                if top_values[i] < -9000: # Descarta acciones infactibles
                    break

                best_index = top_indices[i].item()
                src = int(best_index / (self.input_adapter.S_max - 1))
                r = best_index % (self.input_adapter.S_max - 1)
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

        # Retornamos los hijos y la memoria actualizada
        return children, memory
    
    def eval(self, children, H, memory):
        # Preparación del batch de datos
        batch_data_lists = []
        for state in children:
            data = list(self.input_adapter.input_2_vec(state, H))
            for j in range(len(data)):
                val = data[j]
                data[j] = torch.tensor([val]) if isinstance(val, (int, float)) else torch.from_numpy(val).unsqueeze(0)
            batch_data_lists.append(data)

        batch_inputs = [torch.cat(tensors, dim=0) for tensors in zip(*batch_data_lists)]

        # 3. Inferencia en batch usando la memoria del cost_model
        with torch.no_grad():
            stack_embeddings, memory = self.cost_model.encode(*batch_inputs, memory=memory)
            costs = self.cost_model.decode(stack_embeddings, *batch_inputs)

        # Retornamos los costos como lista plana y la memoria actualizada
        return costs.tolist(), memory