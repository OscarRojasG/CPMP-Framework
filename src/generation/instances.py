import os
from settings import INSTANCE_FOLDER
from cpmp.layout import Layout
import random

def generate_stacks(H, S, N):
    stacks = []
    for _ in range(S):
        stacks.append([])

    for j in range(N):
        s = random.randint(0,S-1)
        while len(stacks[s])==H:
            s = random.randint(0,S-1)
        g = N-j
        stacks[s].append(g)

    return stacks

def generate_instance(filepath, H, S, N):
    stacks = generate_stacks(H, S, N)

    with open(filepath, 'w') as f:
        f.write(f"{S} {N}")
        for s in stacks:
            f.write("\n")
            f.write(f"{len(s)} ")
            for g in s:
                f.write(f"{g} ")

def generate_instances(basename, H, S, N, amount, seed=42):
    os.makedirs(INSTANCE_FOLDER / basename, exist_ok=True)
    random.seed(seed)

    for i in range(amount):
        filepath = INSTANCE_FOLDER / basename / f'{basename}-{i}.txt'
        generate_instance(filepath, H, S, N)

def read_instance(file, H):
    with open(file) as f:
        S, C = [int(x) for x in next(f).split()] # read first line
        stacks = []
        for line in f: # read rest of lines
            stack = [int(x) for x in line.split()[1::]]
            #if stack[0] == 0: stack.pop()
            stacks.append(stack)
            
        layout = Layout(stacks,H)
    return layout