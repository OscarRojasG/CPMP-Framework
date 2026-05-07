from abc import ABC, abstractmethod
from cpmp.layout import Layout

class StackFeaturesAdapter(ABC):
    @abstractmethod
    def to_vec(self, layout: Layout, H: int, S_max: int):
        pass