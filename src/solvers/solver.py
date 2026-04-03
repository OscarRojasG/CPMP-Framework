from abc import ABC, abstractmethod
import os
from settings import INSTANCE_FOLDER
from generation.adapters import LayoutDataAdapter, MovesDataAdapter

class Solver(ABC):
    def __init__(self, name):
        self.name = name
        
    def solve(self, instance_file, H, max_steps, layout_adapter: LayoutDataAdapter, moves_adapter: MovesDataAdapter):
        instance_file = INSTANCE_FOLDER / instance_file
        return self.solve_from_path(instance_file, H, max_steps, layout_adapter, moves_adapter)

    @abstractmethod
    def solve_from_path(self, path, H, max_steps, layout_adapter, moves_adapter):
        pass
    
    def solve_from_folder(self, folder, H, max_steps, layout_adapter, moves_adapter):
        solved_arr = []
        steps_arr = []

        for filename in os.listdir(INSTANCE_FOLDER / folder):
            filepath = os.path.join(INSTANCE_FOLDER / folder, filename)
            solved, steps = self.solve_from_path(filepath, H, max_steps, layout_adapter, moves_adapter)
            solved_arr.append(solved)
            steps_arr.append(steps)

        return solved_arr, steps_arr