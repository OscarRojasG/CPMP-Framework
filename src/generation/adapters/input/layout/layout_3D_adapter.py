import numpy as np
from generation.adapters.input.layout.layout_adapter import LayoutAdapter

class Layout3DAdapter(LayoutAdapter):
    def __init__(self):
        super().__init__()

    def input_2_vec(self, layout, H, S_max=10, H_max=12):
        stacks_matrix = []
        
        # Obtenemos todos los valores para normalizar
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        # 1. Padding en la dimensión de ALTURA (Columnas de la matriz)
        for stack in layout.stacks:
            normalized_stack = [val / max_val for val in stack]
            padding_size = H_max - len(normalized_stack)
            # Aseguramos que no exceda H_max si el stack es más grande
            padded_stack = normalized_stack[:H_max] + [-1] * max(0, padding_size)
            stacks_matrix.append(padded_stack)
        
        # 2. Padding en la dimensión de STACKS (Filas de la matriz)
        # Calculamos cuántas filas vacías (llenas de -1) necesitamos agregar
        num_current_stacks = len(stacks_matrix)
        stacks_to_add = S_max - num_current_stacks
        
        if stacks_to_add > 0:
            # Creamos filas de tamaño H_max rellenas con -1
            padding_rows = [[-1] * H_max for _ in range(stacks_to_add)]
            stacks_matrix.extend(padding_rows)
        else:
            # Si hay más stacks de los permitidos, recortamos a S_max
            stacks_matrix = stacks_matrix[:S_max]
            
        return (np.array(stacks_matrix, dtype=np.float32), )