from abc import ABC, abstractmethod
import os
from settings import INSTANCE_FOLDER
from cpmp.layout import read_file

class Solver(ABC):
    def __init__(self, name):
        self.name = name
        
    def solve(self, instance_file, H, max_steps):
        instance_path = INSTANCE_FOLDER / instance_file
        layout = read_file(instance_path, H)
        return self.solve_from_layouts([layout], H, max_steps)[0]

    @abstractmethod
    def solve_from_layouts(self, layout, H, max_steps):
        pass
    
    def solve_from_folder(self, folder, H, max_steps):
        layouts = []
        for filename in os.listdir(INSTANCE_FOLDER / folder):
            filepath = os.path.join(INSTANCE_FOLDER / folder, filename)
            layouts.append(read_file(filepath, H))
        
        return self.solve_from_layouts(layouts, H, max_steps)