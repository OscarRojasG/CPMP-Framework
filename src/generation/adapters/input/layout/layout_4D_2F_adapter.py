import numpy as np
from generation.adapters.input.layout.layout_adapter import LayoutAdapter

class Layout4D2FAdapter(LayoutAdapter):
    def __init__(self):
        super().__init__()

    def input_2_vec(self, layout, H, H_max=12):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        for i in range(len(layout.stacks)):
            stack = []
            
            # Variables para rastrear el estado de bloqueo
            is_blocked = False
            prev_val = None

            for j in range(len(layout.stacks[i])):
                current_val = layout.stacks[i][j]
                normalized_c = current_val / max_val
                
                # Lógica de Bloqueo (Definición 1)
                # 1. El primero (j == 0) nunca está bloqueado inicialmente.
                # 2. Si ya está bloqueado un nivel superior, el resto hacia abajo también.
                # 3. Si el valor actual es mayor que el anterior, se bloquea.
                if j > 0:
                    if is_blocked or current_val > prev_val:
                        is_blocked = True
                
                # Asignar 1 si está bloqueado, 0 de lo contrario
                blocked_val = 1.0 if is_blocked else 0.0
                
                stack.append([normalized_c, blocked_val])
                prev_val = current_val # Actualizamos para la siguiente iteración
            
            # Padding: Ahora cada elemento es un par [val, blocked]
            # Usamos [-1, -1] para mantener la consistencia con tu código
            padding_size = H_max - len(stack)
            padded_stack = stack + [[-1.0, -1.0]] * padding_size
            stacks_matrix.append(padded_stack)

        return (np.array(stacks_matrix, dtype=np.float32), )