from settings import INSTANCE_FOLDER, DATA_FOLDER, FEG_PATH
import subprocess
from generation.instances import read_instance
import copy
import os
import h5py
import numpy as np
from generation.adapters import *

def greedy(layout, H, max_steps):
    filepath = INSTANCE_FOLDER / "tmp.txt"
    lay2file(layout, filename=filepath)

    result = subprocess.run(
        [FEG_PATH, str(H), filepath, "1.2", str(max_steps), "0", "--no-assignement", "2"],
        check=True,
        text=True,
        capture_output=True
    )
    output_str = result.stdout.split('\t')[0].strip()
    if not output_str.isdigit():
        return float('inf')

    return int(output_str)

def lay2file(layout, filename):
    S = layout.stacks

    with open(filename, "w") as f:
        num_sublists = len(S)
        sum_lengths = sum(len(sublist) for sublist in S)
        f.write(f"{num_sublists} {sum_lengths}\n")
        for sublist in S:
            f.write(str(len(sublist)) +" " + " ".join(str(x) for x in sublist) + "\n")

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
    best_moves = []
    min_cost = float('inf')

    for (i, j) in moves:
        lay_copy = copy.deepcopy(layout)
        lay_copy.move(i, j)
        cost = greedy(lay_copy, H, max_steps)

        if cost < min_cost:
            min_cost = cost
            best_moves = [(i, j)]
        elif cost == min_cost:
            # Si hay empates en la jugada óptima, guarda ambas
            best_moves.append((i, j))

    return best_moves, cost

def generate_data_from_file(filepath, H, max_steps, layout_adapter: LayoutDataAdapter, moves_adapter: MovesDataAdapter):
    layout = read_instance(filepath, H)
    if layout.unsorted_stacks == 0: return None

    layout_vec = layout_adapter.layout_2_vec(layout, H)
    S = len(layout.stacks)

    best_moves, cost = get_best_moves(layout, H, max_steps)
    moves_vec = moves_adapter.moves_2_vec(best_moves, S)

    return layout_vec, moves_vec, cost

def generate_data(folder, H, max_steps, layout_adapter: LayoutDataAdapter, moves_adapter: MovesDataAdapter):
    costs = []

    for input_filename in os.listdir(INSTANCE_FOLDER / folder):
        filepath = os.path.join(INSTANCE_FOLDER / folder, input_filename)
        result = generate_data_from_file(filepath, H, max_steps, layout_adapter, moves_adapter)
        
        if result is None:
            continue

        layout_vec, moves_vec, cost = result
        layout_adapter.add(layout_vec)
        moves_adapter.add(moves_vec)
        costs.append(cost)

    layout_data = layout_adapter.get()
    moves_data = moves_adapter.get()
    data = {**layout_data, **moves_data}

    output_path = DATA_FOLDER / f"{folder}.data"

    with h5py.File(output_path, "w") as f:
        keys_order = [k for k in data.keys() if k != 'C']
        f.attrs['key_order'] = [k for k in keys_order]

        for key in data:
            f.create_dataset(key, data=data[key])
        f.create_dataset("C", data=np.stack(costs, dtype=np.int32))

    print(f"Datos guardados en: {output_path} (Tamaño {layout_adapter.count()})")