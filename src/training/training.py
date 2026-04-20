from torch.utils.data import random_split, DataLoader
import torch
import os
import copy
import json
from settings import MODELS_FOLDER, HYPERPARAMETERS_FOLDER
from torch.amp import GradScaler, autocast
from training.metrics import *
import random
from generation.data import generate_data_rl
from preprocessing.dataset import load_dataset
import torch.multiprocessing as mp
import numpy as np
    
class ModelScorer:
    def __init__(self, model):
        self.model = model
        self.best_models = {}

    def update_best_models(self, epoch, val_metrics: EpochMetrics):
        for metric in val_metrics.metrics:
            sign = 1 if metric.maximize else -1
            score = sign * val_metrics.metrics[metric][-1]

            if metric in self.best_models and score < self.best_models[metric]["score"]: continue

            if metric not in self.best_models:
                self.best_models[metric] = {}
                
            self.best_models[metric]["score"] = score
            self.best_models[metric]["weights"] = copy.deepcopy(self.model.state_dict())
            self.best_models[metric]["epoch"] = epoch

    def print_best_scores(self):
        print("Mejores modelos por métrica:")
        for metric in self.best_models:
            sign = 1 if metric.maximize else -1
            print(f"    {metric.name}: {metric.format(sign * self.best_models[metric]['score'])} (Epoch {self.best_models[metric]['epoch']})")
        
    def print_best_score(self, metric):
        sign = 1 if metric.maximize else -1
        print(f"Mejor modelo ({metric.name}): {metric.format(sign * self.best_models[metric]['score'])} (Epoch {self.best_models[metric]['epoch']})")
    
    def get_best_weights(self):
        return {metric.name: self.best_models[metric]["weights"] for metric in self.best_models}
    
    def get_best_weights_by_metric(self, metric):
        return self.best_models[metric]["weights"]
    
    def get_last_update_epoch(self, metric):
        return self.best_models[metric]["epoch"]
    
    
def train_epoch(model, train_loader, optimizer, loss_function, metrics, device, scaler):
    model.train()

    for *inputs, y_batch in train_loader:
        inputs = [i.to(device, non_blocking=True) for i in inputs]
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Autocast para precisión mixta (FP16)
        with autocast(device.type):
            logits = model(*inputs)
            loss = loss_function.step(logits, y_batch)
            for metric in metrics: metric.step(logits, y_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    loss = loss_function.compute()
    values = [metric.compute() for metric in metrics]

    return loss, values

def val_epoch(model, val_loader, loss_function, metrics, device):
    model.eval()

    with torch.no_grad(), autocast(device.type):
        for batch in val_loader:
            *inputs, y_batch = [i.to(device, non_blocking=True) for i in batch]
            logits = model(*inputs)
            loss = loss_function.step(logits, y_batch)
            for metric in metrics: metric.step(logits, y_batch)

    loss = loss_function.compute()
    values = [metric.compute() for metric in metrics]

    return loss, values

def _train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_function, print_epoch_results, model_scorer, patience, metrics, device): 
    num_workers = os.cpu_count()
    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2,
        persistent_workers=True
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = GradScaler(device.type)

    train_metrics = EpochMetrics()
    val_metrics = EpochMetrics()

    for epoch in range(1, epochs+1):
        loss, values = train_epoch(model, train_loader, optimizer, loss_function, metrics, device, scaler)
        train_metrics.add_value(loss_function, loss)
        for i, value in enumerate(values):
            train_metrics.add_value(metrics[i], value)

        loss, values = val_epoch(model, test_loader, loss_function, metrics, device)
        val_metrics.add_value(loss_function, loss)
        for i, value in enumerate(values):
            val_metrics.add_value(metrics[i], value)

        print_epoch_results(epoch, train_metrics, val_metrics)
        model_scorer.update_best_models(epoch, val_metrics)

        if epoch - model_scorer.get_last_update_epoch(loss_function) > patience:
            print("Early stopping en época", epoch)
            break

    return train_metrics, val_metrics

def generate_sets(dataset, train_size, test_size, seed):
    generator = torch.Generator().manual_seed(seed)
    remaining_size = len(dataset) - train_size - test_size

    train_set, test_set, _ = random_split(
        dataset, 
        [train_size, test_size, remaining_size],
        generator=generator
    )

    return train_set, test_set

def config_training(model, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if torch.backends.mps.is_available() 
                          else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")
    torch.set_num_threads(os.cpu_count())
    model = model.to(device)
    return device

def train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, patience, metrics, device):
    loss_function = CrossEntropyLoss()
    model_scorer = ModelScorer(model)

    def print_epoch_results(epoch: int, train_metrics: EpochMetrics, val_metrics: EpochMetrics):
        print(f'{'\n' if epoch == 1 else ''}Epoch {epoch}/{epochs}')
        print(f"    Average - Train Loss: {loss_function.format(train_metrics.get_last_value(loss_function))}", end='')
        print(f" | Val Loss: {loss_function.format(val_metrics.get_last_value(loss_function))}")

        for i, metric in enumerate(val_metrics.metrics):
            if metric == loss_function: continue
            value = val_metrics.get_last_value(metric)
            print(f'{' | ' if i > 0 else '    '}{metric.name}: {metric.format(value)}', end='')
        print()

    train_metrics, val_metrics = _train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, loss_function, print_epoch_results, model_scorer, patience, metrics, device)
    weights = model_scorer.get_best_weights_by_metric(loss_function)
    model.load_state_dict(weights)
    model_scorer.print_best_score(loss_function)

    return model

def sl_train(model, epochs, dataset, train_size, test_size, batch_size, learning_rate, weight_decay, patience, metrics, seed=42):
    device = config_training(model, seed)
    train_set, test_set = generate_sets(dataset, train_size, test_size, seed)
    return train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, patience, metrics, device)

class DataGenerationConfigRL():
    def __init__(self, instance_set, H, max_steps, layout_adapter_config, moves_adapter_config, num_workers):
        self.instance_set = instance_set
        self.H = H
        self.max_steps = max_steps
        self.layout_adapter_config = layout_adapter_config
        self.moves_adapter_config = moves_adapter_config
        self.num_workers = num_workers

def rl_train(model, iterations, datagen_config, epochs, train_size, test_size, batch_size, learning_rate, weight_decay, patience, metrics, seed=42):
    device = config_training(model, seed)
    dataset_file = "tmp.data"
    last_avg_cost_test = None
    i = 0

    while True:
        if i > 0: print()

        mp.set_start_method('spawn', force=True)
        generate_data_rl(datagen_config.instance_set, 
            datagen_config.H,
            datagen_config.max_steps,
            datagen_config.layout_adapter_config,
            datagen_config.moves_adapter_config,
            model,
            batch_size,
            datagen_config.num_workers,
            output_name=dataset_file)
        
        dataset = load_dataset(dataset_file)
        train_set, test_set = generate_sets(dataset, train_size, test_size, seed)

        dataset._open_file()
        train_costs = dataset.file['C'][sorted(train_set.indices)]
        avg_cost_train = np.mean(train_costs)

        test_costs = dataset.file['C'][sorted(test_set.indices)]
        avg_cost_test = np.mean(test_costs)
        dataset.close()

        print(f"Costo promedio | Train: {avg_cost_train:.2f} | Test: {avg_cost_test:.2f}")

        if last_avg_cost_test:
            current_cost_red = -(avg_cost_test - last_avg_cost_test)
            total_cost_red = -(avg_cost_test - start_avg_cost_test)
            current_gap = current_cost_red / last_avg_cost_test * 100
            total_gap = total_cost_red / start_avg_cost_test * 100

            print(f"Reducción del Costo: {current_cost_red:.2f} (acumulado {total_cost_red:.2f})")
            print(f"Reducción del Gap: {current_gap:.2f}% (acumulado {total_gap:.2f}%)")

            if avg_cost_test >= last_avg_cost_test:
                print(f"Early stopping en iteración {i+1}")
                break
        else:
            start_avg_cost_test = avg_cost_test

        last_avg_cost_test = avg_cost_test
        best_weights = model.state_dict()

        if i == iterations: break
        model = train(model, epochs, train_set, test_set, batch_size, learning_rate, weight_decay, patience, metrics, device)
        i += 1

    if os.path.exists(dataset_file):
        os.remove(dataset_file)

    model.load_state_dict(best_weights)
    return model

def save_model(model, model_name):
    os.makedirs(HYPERPARAMETERS_FOLDER, exist_ok=True)
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'w') as f:
        json.dump(model.hyperparams, f, indent=4)

    os.makedirs(MODELS_FOLDER, exist_ok=True)
    weights = model.state_dict()
    torch.save(weights, str(MODELS_FOLDER / model_name) + ".pth")
    print(f"✅ Modelo guardado en {MODELS_FOLDER / model_name}.pth")

def load_hyperparams(model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        return json.load(f)

def load_model(model_class: object, model_name):
    with open(str(HYPERPARAMETERS_FOLDER / model_name) + ".json", 'r') as f:
        hyperparams = json.load(f)

    model = model_class(**hyperparams)
    model.load_state_dict(torch.load(str(MODELS_FOLDER / model_name) + ".pth", weights_only=True, map_location=torch.device('cpu')), strict=True)
    model.eval()
    return model