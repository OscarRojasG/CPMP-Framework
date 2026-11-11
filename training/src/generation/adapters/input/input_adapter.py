from generation.adapters.data_adapter import DataAdapter
from abc import abstractmethod
from cpmp.layout import Layout

class InputAdapter(DataAdapter):
    def __init__(self, data_keys, S_max, H_max):
        super().__init__(data_keys)
        self.S_max = S_max
        self.H_max = H_max

    @abstractmethod
    def input_2_vec(layout: Layout, H: int):
        pass