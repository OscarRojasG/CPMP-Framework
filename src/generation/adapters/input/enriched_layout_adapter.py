from generation.adapters.input.input_adapter import InputAdapter
import numpy as np
from cpmp.layout import Layout

class EnrichedLayoutAdapter(InputAdapter):
    stack_adapter = None
    extra_data_adapter = None

    def __init__(self, layout_adapter, stack_features_adapter):
        super().__init__({
            "L": np.float32,
            "X": np.float32,
            "S": np.int32,
            "H": np.int32
        })
        self.layout_adapter = layout_adapter()
        self.stack_features_adapter = stack_features_adapter()

    def input_2_vec(self, layout: Layout, H: int, S_max: int = 10, H_max: int = 12):
        L = self.layout_adapter.input_2_vec(layout, H, S_max, H_max)[0]
        X = self.stack_features_adapter.to_vec(layout, H)
        S = len(layout.stacks)
        return L, X, S, H

    def add(self, layout_data):
        L, X, S, H = layout_data

        self.data['L'].append(L)
        self.data['X'].append(X)
        self.data['S'].append(S)
        self.data['H'].append(H)