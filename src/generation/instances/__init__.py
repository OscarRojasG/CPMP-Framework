import os
from settings import INSTANCE_FOLDER
from cpmp.layout import Layout
from solvers.FRG import FRGSolver
from generation.instances.instance_generator import InstanceGenerator
from generation.instances.generators.full_random import FullRandomGenerator
from generation.instances.generators.random_moves import RandomMovesGenerator
from generation.instances.generators.uniform_cost import UniformCostGenerator


def generate_instances(basename: str, instance_generator: InstanceGenerator, amount: int):
    os.makedirs(INSTANCE_FOLDER / basename, exist_ok=True)
    instances = instance_generator.generate_instances(amount)

    for i, inst in enumerate(instances):
        filepath = INSTANCE_FOLDER / basename / f'{basename}-{i}.txt'
        with open(filepath, 'w') as f:
            f.write(f"{instance_generator.S} {instance_generator.N}")
            for s in inst:
                f.write("\n")
                f.write(f"{len(s)} ")
                for g in s:
                    f.write(f"{g} ")

def read_instance(file, H):
    with open(INSTANCE_FOLDER / file) as f:
        S, C = [int(x) for x in next(f).split()] # read first line
        stacks = []
        for line in f: # read rest of lines
            stack = [int(x) for x in line.split()[1::]]
            #if stack[0] == 0: stack.pop()
            stacks.append(stack)
            
        layout = Layout(stacks,H)
    return layout

def get_cost_distribution(folder, H):
    costs = []
    solver = FRGSolver()

    for file in os.listdir(INSTANCE_FOLDER / folder):
        layout = read_instance(os.path.join(folder, file), H)
        _, cost = solver.solve_from_layout(layout, H, 999999)
        costs.append(cost)

    return costs