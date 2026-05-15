from generation.adapters.input.stack_features.stack_features_adapter import StackFeaturesAdapter
from cpmp.layout import Layout
import numpy as np

class StackFeaturesAdapterV1(StackFeaturesAdapter):
    def to_vec(self, layout: Layout, H: int):
        # 1. Inicializamos la matriz con -1.0 para todas las posiciones (S_max, 3)
        X = np.full((self.S_max, 3), -1.0, dtype=np.float32)

        # 2. Iteramos solo hasta el número de stacks reales
        for i in range(len(layout.stacks)):
            # Calculamos las features para los stacks existentes
            is_sorted = 1.0 if layout.is_sorted_stack(i) else 0.0
            height_ratio = len(layout.stacks[i]) / H
            
            # Evitamos división por cero si el stack está vacío
            if len(layout.stacks[i]) != 0:
                sorted_ratio = layout.sorted_elements[i] / len(layout.stacks[i])
            else:
                sorted_ratio = 1.0

            # Asignamos los valores calculados
            X[i][0] = is_sorted
            X[i][1] = height_ratio
            X[i][2] = sorted_ratio

        return X