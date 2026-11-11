from generation.adapters.output.output_adapter import OutputAdapter
import numpy as np

class CostAdapter(OutputAdapter):
    def __init__(self):
        super().__init__({
            "cost": np.float32
        })
    
    def output_2_vec(self, moves, cost):
        return np.log(cost)
    
    def add(self, output_data):
        cost = output_data
        self.data['cost'].append(cost)