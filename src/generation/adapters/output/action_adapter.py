from generation.adapters.output.output_adapter import OutputAdapter
import numpy as np

class ActionAdapter(OutputAdapter):
    def __init__(self):
        super().__init__({
            "Y": np.int32
        })
    
    def output_2_vec(self, moves, cost, S_max=10):
        Y = np.zeros(S_max*(S_max-1), dtype=np.int32)

        for move in moves:
            src, dst = move[0], move[1]
            # Implementación de la fórmula: A = src * (S - 1) + (dst - [dst > src])
            idx = src * (S_max - 1) + (dst - int(dst > src))
            Y[idx] = 1.0

        return Y
    
    def add(self, output_data):
        self.data['Y'].append(output_data)