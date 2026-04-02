import torch
from solvers.solver import Solver
from settings import INSTANCE_FOLDER
import numpy as np
from cpmp.layout import read_file, layout_to_tensors


class ModelSolver(Solver): 
    def __init__(self, model):
        super().__init__("ModelSolver")
        self.model = model
     
    def solve(self, instance_file, H, max_steps):
        instance_file = INSTANCE_FOLDER / instance_file
        layout = read_file(instance_file, H)
        start_unsorted_stacks = layout.unsorted_stacks
        
        with torch.no_grad():
            while not layout.is_sorted():
                G, P, I, S = layout_to_tensors(layout)

                GT = torch.from_numpy(G).unsqueeze(0)
                PT = torch.from_numpy(P).unsqueeze(0)
                IT = torch.from_numpy(I).unsqueeze(0)
                ST = torch.from_numpy(np.array([S])).unsqueeze(0)
                HT = torch.from_numpy(np.array([H])).unsqueeze(0)    

                logits = self.model(GT, PT, IT, ST, HT)

                best_index = logits.argmax(dim=1).item()
                src = int(best_index / (S-1))
                r = best_index % (S-1)
                dst = r if r < src else r + 1

                layout.move(src, dst)

                if layout.steps >= max_steps:
                    break

        return start_unsorted_stacks, layout.unsorted_stacks