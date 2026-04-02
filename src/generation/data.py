from settings import INSTANCE_FOLDER, DATA_FOLDER, FEG_PATH
import subprocess
from generation.instances import read_instance
import copy
import torch
import os
import h5py
import numpy as np
from cpmp.layout import layout_to_tensors

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

    return best_moves

def moves_to_tensor(moves, S):
    Y = torch.zeros(S*(S-1), dtype=torch.int)

    for move in moves:
        idx = move[0] * (S-1) + move[1]
        Y[idx] = 1

    return Y

def generate_data_from_file(filepath, H, max_steps):
    layout = read_instance(filepath, H)

    G, P, I, S = layout_to_tensors(layout)

    best_moves = get_best_moves(layout, H, max_steps)
    Y = moves_to_tensor(best_moves, S)

    return G, P, I, S, Y

def generate_data(folder, H, max_steps):
    all_G, all_P, all_I, all_S, all_H, all_Y = [], [], [], [], [], []

    for input_filename in os.listdir(INSTANCE_FOLDER / folder):
        filepath = os.path.join(INSTANCE_FOLDER / folder, input_filename)
        G, P, I, S, Y = generate_data_from_file(filepath, H, max_steps)

        all_G.append(G)
        all_P.append(P)
        all_I.append(I)
        all_S.append(S)
        all_H.append(H)
        all_Y.append(Y)

    data_G = np.stack(all_G, dtype=np.int32)
    data_P = np.stack(all_P, dtype=np.int32)
    data_I = np.stack(all_I, dtype=np.int32)
    data_S = np.stack(all_S, dtype=np.int32)
    data_H = np.stack(all_H, dtype=np.int32)
    data_Y = np.stack(all_Y, dtype=np.int32)

    output_path = DATA_FOLDER / f"{folder}.data"

    with h5py.File(output_path, "w") as f:
        f.create_dataset("G", data=data_G)
        f.create_dataset("P", data=data_P)
        f.create_dataset("I", data=data_I)
        f.create_dataset("S", data=data_S)
        f.create_dataset("H", data=data_H)
        f.create_dataset("Y", data=data_Y)

    print(f"Datos guardados en: {output_path}")
    