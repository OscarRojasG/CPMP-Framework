import torch
import copy

class PyTorchSearchNode:
    def __init__(self, layout, H, input_adapter, branch_model, value_model=None, visited_memory=None):
        self.layout = layout
        self.H = H
        self.input_adapter = input_adapter
        self.branch_model = branch_model
        self.value_model = value_model
        
        # Referencia al diccionario compartido que rastrea los estados visitados
        self.visited_memory = visited_memory if visited_memory is not None else {}
        
        self.stacks = len(layout.stacks)
        self.move_list = []

        self.dead_end_stack = []

    def is_complete(self):
        return self.layout.is_sorted()

    def apply(self, move):
        src = int(move / (self.input_adapter.S_max - 1))
        r = move % (self.input_adapter.S_max - 1)
        dst = r if r < src else r + 1

        self.layout.move(src, dst)
        self.move_list.append(move)
        
        # 1. Registramos el estado actual y su profundidad
        current_state = tuple(tuple(stack) for stack in self.layout.stacks)
        depth = len(self.move_list)
        
        # 2. Control limpio de ciclos: ¿Ya visitamos este layout exacto a una profundidad menor o igual?
        if current_state in self.visited_memory and self.visited_memory[current_state] <= depth:
            self.dead_end_stack.append(True)
        else:
            self.visited_memory[current_state] = depth
            self.dead_end_stack.append(False)

    def undo_last_move(self):
        move = self.move_list.pop()
        self.dead_end_stack.pop() # Evitamos el envenenamiento del padre
        
        src = int(move / (self.input_adapter.S_max - 1))
        r = move % (self.input_adapter.S_max - 1)
        dst = r if r < src else r + 1

        self.layout.undo_move(src, dst)

    def get_illegal_moves(self):
        # Restaurada a su versión pura de reglas físicas
        illegal_moves = []
        last_dst = -1
        
        if len(self.move_list) > 0:
            last_move = self.move_list[-1]
            last_src = int(last_move / (self.input_adapter.S_max - 1))
            last_r = last_move % (self.input_adapter.S_max - 1)
            last_dst = last_r if last_r < last_src else last_r + 1

        total_actions = self.input_adapter.S_max * (self.input_adapter.S_max - 1)
        
        for move in range(total_actions):
            src = int(move / (self.input_adapter.S_max - 1))
            r = move % (self.input_adapter.S_max - 1)
            dst = r if r < src else r + 1

            if src >= self.stacks or dst >= self.stacks:
                illegal_moves.append(move)
                continue

            if len(self.layout.stacks[dst]) == self.H or len(self.layout.stacks[src]) == 0 or last_dst == src:
                illegal_moves.append(move)

        return illegal_moves

    def get_branch_network_prediction(self):
        self.branch_model.eval()
        with torch.no_grad():
            # Preprocesamos usando tu adapter
            data = list(self.input_adapter.input_2_vec(self.layout, self.H))
            for j in range(len(data)):
                val = data[j]
                data[j] = torch.tensor([val]).to(next(self.branch_model.parameters()).device) if isinstance(val, (int, float)) else torch.from_numpy(val).unsqueeze(0).to(next(self.branch_model.parameters()).device)
            
            # Pasamos por la red (asumo que tu modelo devuelve logits)
            # tree_search.py asume probabilidades, así que aplicamos softmax
            stack_embeddings, _ = self.branch_model.encode(*data)
            logits = self.branch_model.decode(stack_embeddings, *data)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        return probs

    def get_lb_network_prediction(self):
        if self.value_model is None:
            return 0.0
            
        self.value_model.eval()
        with torch.no_grad():
            data = list(self.input_adapter.input_2_vec(self.layout, self.H))
            for j in range(len(data)):
                val = data[j]
                data[j] = torch.tensor([val]).to(next(self.value_model.parameters()).device) if isinstance(val, (int, float)) else torch.from_numpy(val).unsqueeze(0).to(next(self.value_model.parameters()).device)
            
            # Asumiendo que tu value_model devuelve un solo valor escalar
            val_out = self.value_model(*data)
            return val_out.item()

    def __deepcopy__(self, memo):
        # Necesario para algoritmos como LDS o WBS que clonan estados
        layout_copy = copy.deepcopy(self.layout)
        ret = PyTorchSearchNode(layout_copy, self.H, self.input_adapter, self.branch_model, self.value_model)
        ret.move_list = copy.deepcopy(self.move_list)
        return ret
    
    def get_cost(self):
        # Si estamos en un callejón sin salida registrado en el stack, disparamos el costo
        if self.dead_end_stack and self.dead_end_stack[-1]:
            return 999999
            
        return len(self.move_list)

    def get_move_list(self):
        return self.move_list