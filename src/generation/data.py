from settings import INSTANCE_FOLDER, DATA_FOLDER
from generation.instances import read_instance
import copy
import os
import h5py
import numpy as np
from generation.adapters import *
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from solvers.FRG import FRGSolver
from solvers.model import ModelSolver
import torch
import random

def get_feasible_moves(layout):
    moves = []
    num_stacks = len(layout.stacks)

    for i in range(num_stacks):
        if len(layout.stacks[i]) > 0:
            for j in range(num_stacks):
                if i != j and len(layout.stacks[j]) < layout.H:
                    moves.append((i, j))

    return moves
    
def get_best_moves(layout, H, max_steps):
    moves = get_feasible_moves(layout)

    lay_copies = []
    for (i, j) in moves:
        lay_copy = copy.deepcopy(layout)
        lay_copy.move(i, j)
        lay_copies.append(lay_copy)

    results = worker_solver.solve_from_layouts(lay_copies, H, max_steps)

    best_moves = []
    min_cost = float('inf')

    for (move, (solved, cost)) in zip(moves, results):
        if not solved: continue

        if cost < min_cost:
            min_cost = cost
            best_moves = [move]
        elif cost == min_cost:
            best_moves.append(move)
            
    return best_moves, min_cost

def generate_data_from_file(filepath):
    layout = read_instance(filepath, worker_H)
    if layout.unsorted_stacks == 0: 
        return None

    layout_vec = worker_la_adapter.layout_2_vec(layout, worker_H)
    S = len(layout.stacks)

    best_moves, cost = get_best_moves(layout, worker_H, worker_max_steps)
    if len(best_moves) == 0:
        return None

    moves_vec = worker_ma_adapter.moves_2_vec(best_moves, S)

    return layout_vec, moves_vec, cost

def generate_data(filepaths, layout_adapter, moves_adapter, init_worker, init_args, num_workers, output_name, verbose=True):
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=init_args
    ) as executor:
        results = list(executor.map(generate_data_from_file, filepaths))

    la_class, *la_args = layout_adapter
    ma_class, *ma_args = moves_adapter
    layout_adapter = la_class(*la_args)
    moves_adapter = ma_class(*ma_args)

    costs = []
    for result in results:
        if result is None:
            continue

        layout_vec, moves_vec, cost = result
        layout_adapter.add(layout_vec)
        moves_adapter.add(moves_vec)
        costs.append(cost)

    layout_data = layout_adapter.get()
    moves_data = moves_adapter.get()
    data = {**layout_data, **moves_data}

    output_path = DATA_FOLDER / f"{output_name}"

    with h5py.File(output_path, "w") as f:
        keys_order = [k for k in data.keys() if k != 'C']
        f.attrs['key_order'] = [k for k in keys_order]

        for key in data:
            f.create_dataset(key, data=data[key])
        f.create_dataset("C", data=np.stack(costs, dtype=np.int32))

    if verbose:
        print(f"Datos guardados en: {output_path} (Tamaño {layout_adapter.count()})")

def init_worker(H, max_steps, layout_adapter_config, moves_adapter_config):
    global worker_la_adapter
    global worker_ma_adapter
    global worker_H
    global worker_max_steps

    la_class, *la_args = layout_adapter_config
    ma_class, *ma_args = moves_adapter_config
    worker_la_adapter = la_class(*la_args)
    worker_ma_adapter = ma_class(*ma_args)

    worker_H = H
    worker_max_steps = max_steps

def init_worker_sl(H, max_steps, layout_adapter_config, moves_adapter_config):
    global worker_solver

    init_worker(H, max_steps, layout_adapter_config, moves_adapter_config)
    worker_solver = FRGSolver()

def generate_data_sl(folder, H, max_steps, layout_adapter_config, moves_adapter_config, num_workers, output_name):
    init_args = (H, max_steps, layout_adapter_config, moves_adapter_config)
    instance_files = [os.path.join(INSTANCE_FOLDER / folder, f) for f in os.listdir(INSTANCE_FOLDER / folder)]
    generate_data(instance_files, layout_adapter_config, moves_adapter_config, init_worker_sl, init_args, num_workers, output_name)
    
def init_worker_rl(H, max_steps, model_cls, model_params, weights, layout_adapter_config, moves_adapter_config, batch_size):
    global worker_solver

    torch.set_num_threads(1) 
    torch.set_num_interop_threads(1)

    init_worker(H, max_steps, layout_adapter_config, moves_adapter_config)
    model = model_cls(**model_params)
    model.load_state_dict(weights)
    model.eval()
    worker_solver = ModelSolver(model, worker_la_adapter, batch_size)

def generate_data_rl(instance_files, H, max_steps, layout_adapter_config, moves_adapter_config, model, batch_size, num_workers, output_name):
    model_cls = model.__class__
    model_params = model.hyperparams
    weights = model.state_dict()
    
    init_args = (H, max_steps, model_cls, model_params, weights, layout_adapter_config, moves_adapter_config, batch_size)
    generate_data(instance_files, layout_adapter_config, moves_adapter_config, init_worker_rl, init_args, num_workers, output_name, verbose=False)

def split_instances(folder, p1, p2, seed):
    # 1. Preparación de archivos
    path = INSTANCE_FOLDER / folder
    instance_files = [os.path.join(path, f) for f in os.listdir(path)]
    
    # 2. Mezcla aleatoria reproducible
    random.seed(seed)
    random.shuffle(instance_files)
    
    # 3. Normalización de p1 y p2
    total_p = p1 + p2
    p1_norm = p1 / total_p
    
    # 4. Cálculo del índice de división
    total_files = len(instance_files)
    limit = int(total_files * p1_norm)
    
    # 5. Segmentación (Slicing)
    # list1 toma desde el inicio hasta 'limit'
    # list2 toma desde 'limit' hasta el final (asegurando el uso de todos los archivos)
    list1 = instance_files[:limit]
    list2 = instance_files[limit:]
    
    return list1, list2

# Variable globales
worker_solver = None
worker_la_adapter = None
worker_ma_adapter = None
worker_H = None
worker_max_steps = None