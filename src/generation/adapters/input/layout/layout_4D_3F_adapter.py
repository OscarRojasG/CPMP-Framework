import numpy as np
from generation.adapters.input.layout.layout_adapter import LayoutAdapter

class Layout4D3FAdapter(LayoutAdapter):
    def __init__(self):
        super().__init__()

    def input_2_vec(self, layout, H, H_max=12):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        for i in range (len(layout.stacks)):
            stack = []

            for j in range(len(layout.stacks[i])):
                normalized_c = layout.stacks[i][j] / max_val
                valid_top = layout.is_top_valid(i, j)
                valid_bottom = layout.is_bottom_valid(i, j)
                stack.append([normalized_c, valid_top, valid_bottom])
            
            padding_size = H_max - len(stack)
            padded_stack = stack + [[-1, -1, -1]] * padding_size
            stacks_matrix.append(padded_stack)

        return (np.array(stacks_matrix, dtype=np.float32), )