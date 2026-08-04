from settings import INSTANCE_FOLDER, DATA_FOLDER
from instances import read_instance
import copy
import os
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from solvers.model import ModelSolver
import torch

def get_feasible_moves(layout):
    moves = []
    num_stacks = len(layout.stacks)

    for i in range(num_stacks):
        if len(layout.stacks[i]) > 0:
            for j in range(num_stacks):
                if i != j and len(layout.stacks[j]) < layout.H:
                    moves.append((i, j))

    return moves
    
def get_moves_costs(layout, H, max_steps):
    moves = get_feasible_moves(layout)

    lay_copies = []
    for (i, j) in moves:
        lay_copy = copy.deepcopy(layout)
        lay_copy.move(i, j)
        lay_copy.steps = 0
        lay_copies.append(lay_copy)

    results = worker_solver.solve_from_layouts(lay_copies, H, max_steps)
    worker_solver.reset()

    all_moves_costs = []
    min_cost = float('inf')

    for move, (solved, cost) in zip(moves, results):
        if not solved: 
            continue

        if cost + 1 < min_cost:
            min_cost = cost + 1
            
        all_moves_costs.append((move, cost + 1))
            
    return all_moves_costs, min_cost

def generate_data_from_file(filepath):
    layout = read_instance(filepath, worker_H)
    if layout.is_sorted():
        return None

    input_vec = worker_la_adapter.input_2_vec(layout, worker_H)

    moves_costs, best_cost = get_moves_costs(layout, worker_H, worker_max_steps)
    if len(moves_costs) == 0:
        return None

    output_vec = worker_ma_adapter.output_2_vec(moves_costs)

    return input_vec, output_vec, best_cost

def generate_data(filepaths, input_adapter, output_adapter, init_worker, init_args, num_workers, verbose=False):
    if num_workers is None:
        num_workers = os.cpu_count()

    total_files = len(filepaths)
    if verbose:
        print(f"Iniciando generación de datos para {total_files} archivos con {num_workers} workers...")

    results = []
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=init_args
    ) as executor:
        for i, result in enumerate(executor.map(generate_data_from_file, filepaths), 1):
            results.append(result)
            
            if verbose and (i % max(1, total_files // 10) == 0 or i == total_files):
                print(f"Progreso: {i}/{total_files} archivos procesados.")

    la_class, *la_args = input_adapter
    ma_class, *ma_args = output_adapter
    input_adapter = la_class(*la_args)
    output_adapter = ma_class(*ma_args)

    costs = []
    for result in results:
        if result is None:
            continue

        input_vec, output_vec, cost = result
        input_adapter.add(input_vec)
        output_adapter.add(output_vec)
        costs.append(cost)

    input_data = input_adapter.get()
    output_data = output_adapter.get()

    if verbose:
        print("Generación de datos finalizada con éxito.")
    return input_data, output_data, costs

def save_data(input_data, output_data, costs, output_name): 
    output_path = DATA_FOLDER / f"{output_name}"

    with h5py.File(output_path, "w") as f:
        g_input = f.create_group("input")
        g_output = f.create_group("output")

        input_keys = list(input_data.keys())
        for key in input_keys:
            g_input.create_dataset(key, data=input_data[key])
        g_input.attrs['key_order'] = [k for k in input_keys]

        output_keys = list(output_data.keys())
        for key in output_keys:
            g_output.create_dataset(key, data=output_data[key])
        g_output.attrs['key_order'] = [k for k in output_keys]

        f.create_dataset("C", data=np.stack(costs, dtype=np.int32))

    print(f"Datos guardados en: {output_path} (Tamaño {len(costs)})")

def init_worker(H, max_steps, input_adapter_config, output_adapter_config):
    global worker_la_adapter
    global worker_ma_adapter
    global worker_H
    global worker_max_steps

    la_class, *la_args = input_adapter_config
    ma_class, *ma_args = output_adapter_config
    worker_la_adapter = la_class(*la_args)
    worker_ma_adapter = ma_class(*ma_args)

    worker_H = H
    worker_max_steps = max_steps

def init_worker_solver(H, max_steps, input_adapter_config, output_adapter_config, solver_config):
    global worker_solver

    init_worker(H, max_steps, input_adapter_config, output_adapter_config)

    solver_class, *solver_args = solver_config
    worker_solver = solver_class(*solver_args)

def generate_data_solver(folder, H, max_steps, input_adapter_config, output_adapter_config, solver_config, num_workers, output_name_prefix=None):
    init_args = (H, max_steps, input_adapter_config, output_adapter_config, solver_config)
    
    folder_path = INSTANCE_FOLDER / folder
    instance_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
    output_name = f"{folder}.data"
    if output_name_prefix:
        output_name = f"{output_name_prefix}_{output_name}"
    
    input_data, output_data, costs = generate_data(
        instance_files, 
        input_adapter_config, 
        output_adapter_config, 
        init_worker_solver, 
        init_args, 
        num_workers,
        verbose=True
    )
    
    save_data(input_data, output_data, costs, output_name)
    
def init_worker_model(H, max_steps, model_cls, model_params, weights, input_adapter_config, output_adapter_config, batch_size):
    global worker_solver

    torch.set_num_threads(1) 
    torch.set_num_interop_threads(1)

    init_worker(H, max_steps, input_adapter_config, output_adapter_config)
    model = model_cls(**model_params)
    model.load_state_dict(weights)
    model.eval()
    worker_solver = ModelSolver(model, worker_la_adapter, batch_size)

def generate_data_model(folder, H, max_steps, input_adapter_config, output_adapter_config, model, batch_size, num_workers, output_name, verbose=False):
    model_cls = model.__class__
    model_params = model.hyperparams
    weights = model.state_dict()
    
    folder_path = INSTANCE_FOLDER / folder
    instance_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
    init_args = (H, max_steps, model_cls, model_params, weights, input_adapter_config, output_adapter_config, batch_size)
    input_data, output_data, costs = generate_data(
        instance_files, 
        input_adapter_config, 
        output_adapter_config, 
        init_worker_model, 
        init_args, 
        num_workers, 
        verbose
    )
    
    save_data(input_data, output_data, costs, output_name)

# Variables globales
worker_solver = None
worker_la_adapter = None
worker_ma_adapter = None
worker_H = None
worker_max_steps = None