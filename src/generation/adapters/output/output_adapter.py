from generation.adapters.data_adapter import DataAdapter
from abc import abstractmethod

class OutputAdapter(DataAdapter):
    def __init__(self, data_keys):
        super().__init__(data_keys)

    @abstractmethod
    def output_2_vec(moves, cost, S_max):
        pass