import numpy as np
from generation.adapters.input.input_adapter import InputAdapter

class LayoutAdapter(InputAdapter):
    def __init__(self, S_max, H_max):
        super().__init__({
            "S": np.float32
        }, S_max, H_max)

    def add(self, layout_data):
        S_matrix = layout_data[0]
        self.data['S'].append(S_matrix)