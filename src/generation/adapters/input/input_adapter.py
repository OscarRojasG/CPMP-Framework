from generation.adapters.data_adapter import DataAdapter
from abc import abstractmethod
from cpmp.layout import Layout

class InputAdapter(DataAdapter):
    def __init__(self, data_keys):
        super().__init__(data_keys)

    @abstractmethod
    def input_2_vec(layout: Layout, H: int, S_max: int, H_max: int):
        pass