from abc import ABC, abstractmethod

class Solver(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def solve(self, instance_file, max_steps):
        pass