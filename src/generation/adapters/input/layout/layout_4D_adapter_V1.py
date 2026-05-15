import numpy as np
from generation.adapters.input.layout.layout_adapter import LayoutAdapter

class Layout4DAdapterV1(LayoutAdapter):
    def __init__(self, S_max, H_max):
        super().__init__(S_max, H_max)

    def input_2_vec(self, layout, H):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        # 1. Padding en la dimensión de ALTURA (H_max)
        for i in range(len(layout.stacks)):
            stack = []

            for j in range(len(layout.stacks[i])):
                normalized_c = layout.stacks[i][j] / max_val
                valid_top = layout.is_top_valid(i, j)
                valid_bottom = layout.is_bottom_valid(i, j)
                # Cada celda tiene 3 valores
                stack.append([normalized_c, float(valid_top), float(valid_bottom)])
            
            # Aplicamos padding a la altura del stack actual
            padding_size = self.H_max - len(stack)
            # Recortamos por seguridad y rellenamos con [-1, -1, -1]
            padded_stack = stack[:self.H_max] + [[-1.0, -1.0, -1.0]] * max(0, padding_size)
            stacks_matrix.append(padded_stack)

        # 2. Padding en la dimensión de STACKS (S_max)
        num_current_stacks = len(stacks_matrix)
        stacks_to_add = self.S_max - num_current_stacks
        
        if stacks_to_add > 0:
            # Creamos stacks vacíos de tamaño (H_max, 3)
            empty_stack = [[-1.0, -1.0, -1.0]] * self.H_max
            for _ in range(stacks_to_add):
                stacks_matrix.append(empty_stack)
        else:
            # Si excediera el máximo, recortamos la cantidad de stacks
            stacks_matrix = stacks_matrix[:self.S_max]

        # La salida tendrá un shape de (S_max, H_max, 3)
        return (np.array(stacks_matrix, dtype=np.float32), )