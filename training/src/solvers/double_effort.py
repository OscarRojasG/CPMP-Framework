from solvers.solver import Solver
import signal
import math
import time

# Definimos una excepción personalizada para el timeout
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

class DoubleEffortSolver(Solver): 
    def __init__(self, solver, max_time):
        super().__init__("DSE " + solver.name)
        self.solver = solver
        self.start_w = self.solver.w
        self.max_time = max_time

    def solve_from_layouts(self, layouts, H, max_steps):
        results = []
        for layout in layouts:
            r = self.solve_from_layout(layout, H, max_steps)
            r = [r[0], r[1]]
            results.append(r)

        return results

    def solve_from_layout(self, layout, H, max_steps):
        t0 = time.perf_counter()
        solved = False
        best_steps = float("inf")

        self.solver.w = self.start_w
        
        signal.signal(signal.SIGALRM, timeout_handler)

        try:
            while True:
                t_elapsed = time.perf_counter() - t0
                remaining = self.max_time - t_elapsed

                if remaining <= 0:
                    break

                # setitimer usa (reloj, tiempo_inicial, intervalo_repeticion)
                # El segundo parámetro es el que nos interesa
                signal.setitimer(signal.ITIMER_REAL, remaining)

                try:
                    res_solved, steps, _ = self.solver.solve_from_layout(layout, H, max_steps)
                    
                    # Desactivar timer
                    signal.setitimer(signal.ITIMER_REAL, 0)

                    if res_solved and steps < best_steps:
                        solved = True
                        best_steps = steps

                except TimeoutException:
                    break 

                self.solver.w = round(self.solver.w * math.sqrt(2)) if self.solver.w > 1 else 2

        finally:
            # Limpieza final por seguridad
            signal.setitimer(signal.ITIMER_REAL, 0)

        return solved, best_steps