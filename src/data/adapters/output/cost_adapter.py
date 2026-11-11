from data.adapters.output.output_adapter import OutputAdapter
import numpy as np

class CostAdapter(OutputAdapter):
    def __init__(self):
        super().__init__({
            "cost": np.float32
        })
    
    def output_2_vec(self, moves_cost):
        min_cost = min(cost for move, cost in moves_cost)
        # Target of the paper: y = ln(c_min + 1), the same as in the archived dataset (cost.data)
        return np.log(min_cost + 1)
    
    def add(self, output_data):
        cost = output_data
        self.data['cost'].append(cost)