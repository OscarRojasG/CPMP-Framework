from settings import INSTANCE_FOLDER, FRG_PATH
from solvers.solver import Solver
from generation.adapters import *
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial


class FRGSolver(Solver): 
    def __init__(self):
        super().__init__("FRG")

    @staticmethod
    def solve_from_layout(layout, H, max_steps):
        pid = os.getpid()
        filepath = INSTANCE_FOLDER / f"tmp_{pid}.txt"

        try:
            FRGSolver.lay2file(layout, filepath)

            result = subprocess.run(
                [FRG_PATH, str(H), filepath, "1.2", str(max_steps), "0", "--no-assignement", "2"],
                check=True,
                text=True,
                capture_output=True
            )
            
            output_str = result.stdout.split('\t')[0].strip()
            if not output_str.isdigit():
                return False, float('inf')

            return True, int(output_str)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    @staticmethod
    def lay2file(layout, filename):
        S = layout.stacks

        with open(filename, "w") as f:
            num_sublists = len(S)
            sum_lengths = sum(len(sublist) for sublist in S)
            f.write(f"{num_sublists} {sum_lengths}\n")
            for sublist in S:
                f.write(str(len(sublist)) +" " + " ".join(str(x) for x in sublist) + "\n")

    def solve_from_layouts(self, layouts, H, max_steps):
        results = []
        for layout in layouts:
            r = self.solve_from_layout(layout, H, max_steps)
            results.append(r)

        return results