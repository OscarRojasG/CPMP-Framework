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

def get_feasible_moves(layout):
    moves = []
    num_stacks = len(layout.stacks)

    for i in range(num_stacks):
        if len(layout.stacks[i]) > 0:
            for j in range(num_stacks):
                if i != j and len(layout.stacks[j]) < layout.H:
                    moves.append((i, j))

    return moves
    
def get_best_moves(layout, H, max_steps, solver):
    moves = get_feasible_moves(layout)

    lay_copies = []
    for (i, j) in moves:
        lay_copy = copy.deepcopy(layout)
        lay_copy.move(i, j)
        lay_copies.append(lay_copy)

    results = solver.solve_from_layouts(lay_copies, H, max_steps)

    best_moves = []
    min_cost = float('inf')

    for solved, cost in results:
        if not solved: continue

        if cost < min_cost:
            min_cost = cost
            best_moves = [(i, j)]
        elif cost == min_cost:
            # Si hay empates en la jugada óptima, guarda ambas
            best_moves.append((i, j))

    return best_moves, min_cost

def generate_data_from_file(filepath, H, max_steps, layout_adapter, moves_adapter, solver):
    layout = read_instance(filepath, H)
    if layout.unsorted_stacks == 0: 
        return None

    layout_vec = layout_adapter.layout_2_vec(layout, H)
    S = len(layout.stacks)

    best_moves, cost = get_best_moves(layout, H, max_steps, solver)
    if len(best_moves) == 0:
        return None

    moves_vec = moves_adapter.moves_2_vec(best_moves, S)

    return layout_vec, moves_vec, cost

def generate_data(folder, H, max_steps, layout_adapter, moves_adapter, solver, output_name):
    filepaths = [os.path.join(INSTANCE_FOLDER / folder, f) for f in os.listdir(INSTANCE_FOLDER / folder)]

    costs = []
    for filepath in filepaths:
        result = generate_data_from_file(filepath, H, max_steps, layout_adapter, moves_adapter, solver)
        if result is None:
            continue

        layout_vec, moves_vec, cost = result
        layout_adapter.add(layout_vec)
        moves_adapter.add(moves_vec)
        costs.append(cost)

    layout_data = layout_adapter.get()
    moves_data = moves_adapter.get()
    data = {**layout_data, **moves_data}

    if output_name is None:
        output_path = DATA_FOLDER / f"{folder}.data"
    else:
        output_path = DATA_FOLDER / f"{output_name}.data"

    with h5py.File(output_path, "w") as f:
        keys_order = [k for k in data.keys() if k != 'C']
        f.attrs['key_order'] = [k for k in keys_order]

        for key in data:
            f.create_dataset(key, data=data[key])
        f.create_dataset("C", data=np.stack(costs, dtype=np.int32))

    print(f"Datos guardados en: {output_path} (Tamaño {layout_adapter.count()})")

def generate_data_sl(folder, H, max_steps, layout_adapter, moves_adapter, output_name=None):
    solver = FRGSolver()
    generate_data(folder, H, max_steps, layout_adapter, moves_adapter, solver, output_name)
    
def generate_data_rl(folder, H, max_steps, layout_adapter, moves_adapter, model, batch_size, output_name=None):
    solver = ModelSolver(model, layout_adapter, batch_size)
    generate_data(folder, H, max_steps, layout_adapter, moves_adapter, solver, output_name)