from generation.adapters.input.input_adapter import InputAdapter
import numpy as np
from cpmp.layout import Layout

class EnrichedLayoutAdapter(InputAdapter):
    stack_adapter = None
    extra_data_adapter = None

    def __init__(self, layout_adapter, stack_features_adapter):
        super().__init__({
            "S": np.float32,
            "X": np.float32,
            "H": np.int32
        })
        self.layout_adapter = layout_adapter()
        self.stack_features_adapter = stack_features_adapter()

    def input_2_vec(self, layout: Layout, H: int, H_max: int = 12):
        S = self.layout_adapter.input_2_vec(layout, H, H_max)[0]
        X = self.stack_features_adapter.to_vec(layout, H)
        return S, X, H

    def add(self, layout_data):
        S_matrix, X, H = layout_data

        self.data['S'].append(S_matrix)
        self.data['X'].append(X)
        self.data['H'].append(H)