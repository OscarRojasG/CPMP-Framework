import os
from settings import INSTANCE_FOLDER
from cpmp.layout import Layout
import random
from abc import ABC, abstractmethod
from solvers.FRG import FRGSolver

class InstanceGenerator(ABC):
    def __init__(self, H, S, N, seed):
        self.H = H
        self.S = S
        self.N = N
        self.seed = seed
        random.seed(seed)

        self.instances = []
        self.instance_set = set()

    def generate_stacks(self, H, S, N, sorted):
        stacks = []
        for _ in range(S):
            stacks.append([])

        for j in range(N):
            s = random.randint(0,S-1)
            while len(stacks[s])==H:
                s = random.randint(0,S-1)
            if sorted:
                g = N - j
            else:
                g = random.randint(1,N)
            stacks[s].append(g)

        return stacks
    
    def add_instance(self, instance):
        instance_hash = tuple(tuple(stack) for stack in instance)
        if tuple(instance_hash) in self.instance_set: return

        layout = Layout(instance, self.H)
        if layout.is_sorted(): return
        
        self.instances.append(instance)
        self.instance_set.add(instance_hash)

    @abstractmethod
    def generate_instances(self, amount):
        pass

class FullRandomGenerator(InstanceGenerator):
    def generate_instances(self, amount):
        while len(self.instances) < amount:
            stacks = self.generate_stacks(self.H, self.S, self.N, sorted=False)
            self.add_instance(stacks)
        return self.instances
    
class RandomMovesGenerator(InstanceGenerator):
    def __init__(self, H, S, N, r, seed):
        super().__init__(H, S, N, seed)
        self.r = r

    def generate_instances(self, amount):
        while len(self.instances) < amount:
            stacks = self.generate_stacks(self.H, self.S, self.N, sorted=True)
            stacks = self.random_moves(stacks, self.H, self.r)
            self.add_instance(stacks)
        return self.instances
    
    def random_moves(self, stacks, H, r):
        last_move = (None, None)
        moves_made = 0
        
        while moves_made < r:
            # 1. Elegir un origen que no esté vacío
            valid_origins = [i for i, s in enumerate(stacks) if len(s) > 0]
            if not valid_origins:
                break  # No hay movimientos posibles
                
            origin_idx = random.choice(valid_origins)
            
            # 2. Elegir un destino que no esté lleno y no sea el origen
            valid_destinations = [
                i for i, s in enumerate(stacks) 
                if i != origin_idx and len(s) < H
            ]
            
            if not valid_destinations:
                continue # Reintentar con otro origen
                
            dest_idx = random.choice(valid_destinations)
            
            # 3. Validar que no anule el movimiento anterior
            # El inverso de (a, b) es (b, a)
            if (dest_idx, origin_idx) == last_move:
                continue
                
            # Ejecutar el movimiento
            container = stacks[origin_idx].pop()
            stacks[dest_idx].append(container)
            
            # Registrar rastro
            last_move = (origin_idx, dest_idx)
            moves_made += 1

        return stacks

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