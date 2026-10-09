from settings import INSTANCE_FOLDER, FRG_PATH
from solvers.solver import Solver
import subprocess
import os
from cpmp.layout import lay2file


class FRGSolver(Solver):
    def __init__(self):
        super().__init__("FRG")

    # El binario reporta pasos relativos al layout recibido (no conoce
    # layout.steps), así que una instancia ordenada vale 0 y no los acumulados
    def _solve_sorted(self, layout):
        return True, 0, 0.0

    def _solve_from_layout(self, layout, H, max_steps):
        output_str = self._run(layout, H, max_steps)[0].split('\t')
        steps_str = output_str[0].strip()
        if not steps_str.isdigit():
            solved = False
            steps = float('inf')
        else:
            solved = True
            steps = int(steps_str)

        time_str = output_str[1].strip()
        return solved, steps, float(time_str)

    # Retorna la secuencia de movimientos (src, dst) de la solución
    def get_moves(self, layout, H, max_steps):
        moves = []
        for line in self._run(layout, H, max_steps)[1:]:
            line = line.split(',')
            moves.append((int(line[0]), int(line[1])))
        return moves

    @staticmethod
    def _run(layout, H, max_steps):
        pid = os.getpid()
        filepath = INSTANCE_FOLDER / f"tmp_{pid}.txt"

        try:
            lay2file(layout, filepath)

            result = subprocess.run(
                [FRG_PATH, str(H), filepath, "1.2", str(max_steps), "0", "--no-assignment", "2"],
                check=True,
                text=True,
                capture_output=True
            )

            return result.stdout.strip().split('\n')
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
