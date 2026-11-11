from evaluators.utils import print_progress_table, stats_to_df
from solvers.double_effort import DoubleEffortSolver
from settings import INSTANCE_FOLDER
import os

def dse_eval(solver_list, folder, H, max_steps, max_time):
    solver_list = [DoubleEffortSolver(solver, max_time) for solver in solver_list]
    solver_stats = {s.name: {'Solved': [], 'Steps': []} for s in solver_list}
    instances = os.listdir(INSTANCE_FOLDER / folder)

    for i, instance in enumerate(instances, start=1):
        for solver in solver_list:
            solved, steps = solver.solve(os.path.join(folder, instance), H, max_steps)

            solver_stats[solver.name]['Solved'].append(solved)
            solver_stats[solver.name]['Steps'].append(steps)
            
        print_progress_table(i, solver_stats)
        
    return stats_to_df(solver_stats, instances)