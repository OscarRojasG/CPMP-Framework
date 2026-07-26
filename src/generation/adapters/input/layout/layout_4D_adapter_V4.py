import numpy as np
from generation.adapters.input.layout.layout_adapter import LayoutAdapter

class Layout4DAdapterV4(LayoutAdapter):
    def __init__(self, S_max, H_max):
        super().__init__(S_max, H_max)

    def input_2_vec(self, layout, H):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        # 1. Procesar stacks existentes y aplicar padding de ALTURA
        for i in range(len(layout.stacks)):
            stack = []
            H_stack = len(layout.stacks[i])

            # Procesamos cada contenedor en el stack actual
            for j in range(H_stack):
                current_val = layout.stacks[i][j]
                normalized_c = current_val / max_val

                stack.append([normalized_c])
            
            # Padding de Altura: Rellenamos con [-1.0] hasta H_max
            padding_size = self.H_max - len(stack)
            padded_stack = stack + [[-1.0]] * max(0, padding_size)
            stacks_matrix.append(padded_stack)

        # 2. Padding de STACKS: Rellenamos con stacks vacíos hasta S_max
        num_current_stacks = len(stacks_matrix)
        stacks_to_add = self.S_max - num_current_stacks
        
        if stacks_to_add > 0:
            empty_stack = [[-1.0]] * self.H_max
            for _ in range(stacks_to_add):
                stacks_matrix.append(empty_stack)
        else:
            stacks_matrix = stacks_matrix[:self.S_max]

        # El resultado será una matriz de dimensiones (S_max, H_max, 1)
        return np.array(stacks_matrix, dtype=np.float32), len(layout.stacks), H