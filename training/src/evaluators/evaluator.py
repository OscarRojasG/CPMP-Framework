from evaluators.utils import print_progress_table, stats_to_df
from settings import INSTANCE_FOLDER
import os

def solver_eval(solver_list, folder, H, max_steps):
    solver_stats = {s.name: {'Solved': [], 'Steps': [], 'Time': []} for s in solver_list}
    instances = os.listdir(INSTANCE_FOLDER / folder)

    for i, instance in enumerate(instances, start=1):
        for solver in solver_list:
            solved, steps, time = solver.solve(os.path.join(folder, instance), H, max_steps)

            solver_stats[solver.name]['Solved'].append(solved)
            solver_stats[solver.name]['Steps'].append(steps)
            solver_stats[solver.name]['Time'].append(time)
            
        print_progress_table(i, solver_stats)
        
    return stats_to_df(solver_stats, instances)