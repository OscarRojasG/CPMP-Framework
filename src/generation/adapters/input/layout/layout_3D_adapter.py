import numpy as np
from generation.adapters.input.layout.layout_adapter import LayoutAdapter

class Layout3DAdapter(LayoutAdapter):
    def __init__(self):
        super().__init__()

    def input_2_vec(self, layout, H, H_max=12):
        stacks_matrix = []
        
        all_vals = [c for s in layout.stacks for c in s]
        max_val = max(all_vals) if all_vals else 1

        for stack in layout.stacks:
            normalized_stack = [val / max_val for val in stack]
            padding_size = H_max - len(normalized_stack)
            padded_stack = normalized_stack + [-1] * padding_size
            stacks_matrix.append(padded_stack)
            
        return (np.array(stacks_matrix, dtype=np.float32), )