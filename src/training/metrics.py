from abc import ABC, abstractmethod
import torch

class EpochMetrics():
    def __init__(self):
        self.metrics = {}

    def add_value(self, metric_cls, value):
        if metric_cls not in self.metrics:
            self.metrics[metric_cls] = []
            
        self.metrics[metric_cls].append(value)

    def get_last_value(self, metric_cls):
        return self.metrics[metric_cls][-1]

class Metric(ABC):
    def __init__(self, name, maximize=True):
        self.name = name
        self.maximize = maximize
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, logits, y):
        pass

    @abstractmethod
    def _compute(self):
        pass

    def compute(self):
        value = self._compute()
        self.reset()
        return value

    def format(self, value):
        return f"{value:.2f}"
    
class Accuracy(Metric):
    def __init__(self, k=1):
        super().__init__("Accuracy" if k == 1 else f"Top-{k} Accuracy")
        self.k = k

    def reset(self):
        self.total_correct = 0
        self.total_samples = 0
    
    def step(self, logits, y):
        batch_size = y.size(0)
        target_indices = y.argmax(dim=-1)
        _, top_k_indices = logits.topk(self.k, dim=1, largest=True, sorted=True)
        correct = top_k_indices.eq(target_indices.view(-1, 1).expand_as(top_k_indices))
        self.total_correct += correct.sum().item()
        self.total_samples += batch_size

    def _compute(self):
        return 100 * self.total_correct / self.total_samples
    
    def format(self, value):
        return f"{value:.2f}%"
    
class CrossEntropyLoss(Metric):
    def __init__(self):
        super().__init__("CrossEntropy", False)

    def reset(self):
        self.total_samples = 0
        self.total_ce = 0
    
    def step(self, logits, y):
        ce = torch.nn.functional.cross_entropy(logits, y)
        batch_size = y.size(0)
        self.total_ce += ce.item() * batch_size 
        self.total_samples += batch_size
        return ce

    def _compute(self):
        return self.total_ce / self.total_samples
    
    def format(self, value):
        return f"{value:.4f}"