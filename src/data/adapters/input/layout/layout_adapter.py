import numpy as np
from data.adapters.input.input_adapter import InputAdapter

class LayoutAdapter(InputAdapter):
    def __init__(self, S_max, H_max):
        super().__init__({
            "L": np.float32,
            "S": np.int32,
            "H": np.int32
        }, S_max, H_max)

    def add(self, layout_data):
        L, S, H = layout_data

        self.data['L'].append(L)
        self.data['S'].append(S)
        self.data['H'].append(H)