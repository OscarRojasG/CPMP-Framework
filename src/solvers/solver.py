from abc import ABC, abstractmethod
import os
from settings import INSTANCE_FOLDER
from cpmp.layout import read_file

class Solver(ABC):
    def __init__(self, name):
        self.name = name
        
    def solve(self, instance_file, H, max_steps):
        instance_path = INSTANCE_FOLDER / instance_file
        layout = read_file(instance_path, H)
        return self.solve_from_layout(layout, H, max_steps)

    # Función para evaluar una instancia a la vez
    # Retorna solved, steps, time
    # Las subclases no la sobrescriben: implementan _solve_from_layout
    def solve_from_layout(self, layout, H, max_steps):
        if layout.is_sorted():
            return self._solve_sorted(layout)
        return self._solve_from_layout(layout, H, max_steps)

    # Función optimizada para varias instancias
    # Retorna solved, steps
    # Las subclases no la sobrescriben: implementan _solve_from_layouts
    def solve_from_layouts(self, layouts, H, max_steps):
        results = [None] * len(layouts)
        pending = []
        for i, layout in enumerate(layouts):
            if layout.is_sorted():
                r = self._solve_sorted(layout)
                results[i] = [r[0], r[1]]
            else:
                pending.append(i)

        if pending:
            pending_results = self._solve_from_layouts([layouts[i] for i in pending], H, max_steps)
            for i, r in zip(pending, pending_results):
                results[i] = r

        return results

    # Instancia ya ordenada: se resuelve con los pasos acumulados del layout
    # Retorna solved, steps, time
    def _solve_sorted(self, layout):
        return True, layout.steps, 0.0

    # Retorna solved, steps, time
    @abstractmethod
    def _solve_from_layout(self, layout, H, max_steps):
        pass

    # Retorna solved, steps
    def _solve_from_layouts(self, layouts, H, max_steps):
        results = []
        for layout in layouts:
            r = self._solve_from_layout(layout, H, max_steps)
            r = [r[0], r[1]]
            results.append(r)

        return results
    
    def solve_from_folder(self, folder, H, max_steps):
        layouts = []
        for filename in os.listdir(INSTANCE_FOLDER / folder):
            filepath = os.path.join(INSTANCE_FOLDER / folder, filename)
            layouts.append(read_file(filepath, H))
        
        return self.solve_from_layouts(layouts, H, max_steps)
    
    def reset(self):
        pass