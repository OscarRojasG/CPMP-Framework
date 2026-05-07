import numpy as np
from generation.adapters.input.layout.layout_adapter import LayoutAdapter

class Layout4D2FV1Adapter(LayoutAdapter):
    def __init__(self):
        super().__init__()

    def input_2_vec(self, layout, H, S_max=10, H_max=12):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        # 1. Procesar stacks existentes y aplicar padding de ALTURA
        for i in range(len(layout.stacks)):
            stack = []
            is_blocked = False
            prev_val = None

            # Procesamos cada contenedor en el stack actual
            for j in range(len(layout.stacks[i])):
                current_val = layout.stacks[i][j]
                normalized_c = current_val / max_val
                
                if j > 0:
                    if is_blocked or current_val > prev_val:
                        is_blocked = True
                
                blocked_val = 1.0 if is_blocked else 0.0
                stack.append([normalized_c, blocked_val])
                prev_val = current_val 
            
            # Padding de Altura: Rellenamos con [-1.0, -1.0] hasta H_max
            padding_size = H_max - len(stack)
            # Recortamos si excede H_max y añadimos padding si falta
            padded_stack = stack[:H_max] + [[-1.0, -1.0]] * max(0, padding_size)
            stacks_matrix.append(padded_stack)

        # 2. Padding de STACKS: Rellenamos con stacks vacíos hasta S_max
        num_current_stacks = len(stacks_matrix)
        stacks_to_add = S_max - num_current_stacks
        
        if stacks_to_add > 0:
            # Creamos stacks vacíos donde cada celda es [-1.0, -1.0]
            empty_stack = [[-1.0, -1.0]] * H_max
            for _ in range(stacks_to_add):
                stacks_matrix.append(empty_stack)
        else:
            # Si hay más stacks de los permitidos, recortamos
            stacks_matrix = stacks_matrix[:S_max]

        # El resultado será una matriz de dimensiones (S_max, H_max, 2)
        return (np.array(stacks_matrix, dtype=np.float32), )