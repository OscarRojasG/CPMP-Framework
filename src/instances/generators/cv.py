from instances.generators.instance_generator import InstanceGenerator
import random

class CVGenerator(InstanceGenerator):
    def __init__(self, H, S, seed):
        super().__init__(H, S, (H-2)*S, seed)

    def gen_cv(self, stacks, tiers):
        total = stacks * tiers
        containers = list(range(1, total + 1))
        random.shuffle(containers)
        ret = []
        for s in range(stacks):
            start_pos = s * tiers
            end_pos = (s + 1) * tiers
            ret.append(containers[start_pos:end_pos])
        return ret

    def generate_instances(self, amount):
        curr_len = len(self.instances)
        while len(self.instances) < amount + curr_len:
            stacks = self.gen_cv(self.S, self.H-2)
            self.add_instance(stacks)
        return self.instances[curr_len:]