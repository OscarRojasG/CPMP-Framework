from abc import ABC, abstractmethod
from cpmp.layout import Layout

class StackFeaturesAdapter(ABC):
    def __init__(self, S_max, H_max):
        self.S_max = S_max
        self.H_max = H_max

    @abstractmethod
    def to_vec(self, layout: Layout, H: int):
        pass