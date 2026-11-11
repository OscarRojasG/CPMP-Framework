from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class Transformer(nn.Module, ABC):
    def __init__(self, **hyperparams):
        torch.manual_seed(42)
        super(Transformer, self).__init__()
        self.hyperparams = hyperparams

    @abstractmethod
    def encode(self, *args, memory=None):
        pass

    @abstractmethod
    def decode(self, *args):
        pass

    def forward(self, *args):
        stack_embeddings, _ = self.encode(*args)
        return self.decode(stack_embeddings, *args)