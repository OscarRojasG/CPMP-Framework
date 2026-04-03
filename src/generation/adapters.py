import numpy as np
from abc import ABC, abstractmethod

class DataAdapter(ABC):
    def __init__(self):
        super().__init__()
        self.data = {}

    @abstractmethod
    def add(self, layout_data):
        pass

    def get(self) -> dict:
        return {
            k: np.stack(v, dtype=np.int32) for k, v in self.data.items()
        }

    def count(self):
        return len(self.data[list(self.data.keys())[0]])

class LayoutDataAdapter(DataAdapter):
    @abstractmethod
    def layout_2_vec(self, layout, H):
        pass

class MovesDataAdapter(DataAdapter):
    @abstractmethod
    def moves_2_vec(self, moves, S):
        pass

class GPIAdapter(LayoutDataAdapter):
    def __init__(self):
        super().__init__()
        self.data = {
            "G": [],
            "P": [],
            "I": [],
            "S": [],
            "H": [],
        }

    def layout_2_vec(self, layout, H):
        G = [] # Valores de grupo
        P = [] # Dónde se ubica el contenedor en su respectiva pila
        I = [] # En qué pila se encuentra el contenedor
        S = len(layout.stacks) # Número de pilas

        for i in range(S):
            for j in range(len(layout.stacks[i])):
                G.append(layout.stacks[i][j])
                P.append(j)
                I.append(i)

        return np.array(G), np.array(P), np.array(I), S, H
    
    def add(self, layout_data):
        G, P, I, S, H = layout_data

        self.data['G'].append(G)
        self.data['P'].append(P)
        self.data['I'].append(I)
        self.data['S'].append(S)
        self.data['H'].append(H)

class StackMatrixAdapter(LayoutDataAdapter):
    def __init__(self):
        super().__init__()
        self.data = {
            "S": []
        }

    def layout_2_vec(self, layout, H):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        for stack in layout.stacks:
            normalized_stack = [val / max_val for val in stack]
            padding_size = H - len(normalized_stack)
            padded_stack = normalized_stack + [-1] * padding_size
            stacks_matrix.append(padded_stack)
            
        return (np.array(stacks_matrix, dtype=np.float32), )

    def add(self, layout_data):
        S_matrix = layout_data[0]
        self.data['S'].append(S_matrix)

    def get(self):
        return {
            k: np.stack(v) for k, v in self.data.items()
        }
    
class DefaultMovesAdapter(MovesDataAdapter):
    def __init__(self):
        super().__init__()
        self.data = {
            "Y": [],
        }
    
    def moves_2_vec(self, moves, S):
        Y = np.zeros(S*(S-1), dtype=np.int32)

        for move in moves:
            src, dst = move[0], move[1]
            # Implementación de la fórmula: A = src * (S - 1) + (dst - [dst > src])
            idx = src * (S - 1) + (dst - int(dst > src))
            Y[idx] = 1.0

        return Y
    
    def add(self, moves_data):
        self.data['Y'].append(moves_data)