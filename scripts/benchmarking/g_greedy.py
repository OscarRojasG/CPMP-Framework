# ==================== Parámetros ====================
# Conjunto de benchmarks a evaluar: "g1", "g2" o "g3"
DATASET = "g1"

# Modelo a cargar (nombre en la carpeta models/)
ACTION_MODEL_NAME = "rl"

# Subcarpetas del benchmark a evaluar ("H-2"-"S")
BENCHS = ["5-5", "5-7", "5-10"]

MAX_STEPS = 100
# ====================================================

# ==============================================================================
# IMPORTANTE: Variables de entorno para evitar bloqueo de CPU en multiprocesamiento
# ==============================================================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

from settings import INSTANCE_FOLDER, EXPERIMENTS_FOLDER
from training.common import load_model
from models.actions.default import CPMPTransformer
from solvers import ModelSolver
from data.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1


# ---------------------------------------------------------
# 1. Caché local para los subprocesos
# ---------------------------------------------------------
worker_solver_cache = None
worker_current_H = None
worker_current_S = None

def get_or_create_solver(H, S):
    """
    Se ejecuta DENTRO de cada worker. Instancia los modelos SOLO
    la primera vez, o si cambian H o S (al pasar a otra carpeta).
    """
    global worker_solver_cache, worker_current_H, worker_current_S

    # Comprobamos si ya tenemos el solver instanciado para estos H y S
    if worker_solver_cache is not None and worker_current_H == H and worker_current_S == S:
        return worker_solver_cache

    # 1. Cargar modelo predictor de acciones
    action_model = load_model(CPMPTransformer, ACTION_MODEL_NAME)

    # 2. Crear el adaptador y el ModelSolver DENTRO del proceso
    input_adapter = EnrichedLayoutAdapter(Layout4DAdapterV1, StackFeaturesAdapterV1, S, H)
    worker_solver_cache = ModelSolver(action_model, input_adapter)

    worker_current_H = H
    worker_current_S = S

    return worker_solver_cache


# ---------------------------------------------------------
# 2. Función del worker (Solo recibe datos ligeros)
# ---------------------------------------------------------
def solve_single_instance(instance_path, H, S, max_steps):
    # Recupera o crea el solver instanciado DENTRO de este proceso pasándole H y S
    solver = get_or_create_solver(H, S)
    solved, steps, t = solver.solve(instance_path, H, max_steps)
    return (solved, steps, t)


# ---------------------------------------------------------
# 3. Bucle principal
# ---------------------------------------------------------
def main():
    base_folder = INSTANCE_FOLDER / "benchmarks" / DATASET

    # Asegurarnos de que el directorio de experimentos exista
    EXPERIMENTS_FOLDER.mkdir(parents=True, exist_ok=True)

    # Usamos CPU al 100%
    MAX_WORKERS = os.cpu_count()

    # Iniciamos el Pool UNA SOLA VEZ para reutilizar procesos
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        for b in BENCHS:
            folder_path = os.path.join(base_folder, b)

            # Extraemos S y H dinámicamente ("5-10" -> S=10, H=5+2=7)
            parts = b.split("-")
            H = int(parts[0]) + 2
            S = int(parts[1])

            instances = sorted(os.listdir(folder_path))
            total = len(instances)

            # Nombre del archivo para esta carpeta en específico
            csv_filename = f"ModelSolver_{DATASET}_{b}.csv"
            csv_filepath = EXPERIMENTS_FOLDER / csv_filename

            print(f"\n--- Iniciando benchmarking para {DATASET}/{b} ({total} instancias en {MAX_WORKERS} procesos paralelos) ---")
            print(f"Guardando resultados en: {csv_filepath}")

            # Abrimos un CSV nuevo para la carpeta actual
            with open(csv_filepath, mode='w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(['Instance', 'Solved', 'Steps'])

                futuros = {}

                # Encolar tareas enviando SOLO la ruta, la altura (H), el número de stacks (S) y max_steps
                for instance_file in instances:
                    instance_path = os.path.join(folder_path, instance_file)
                    futuro = executor.submit(solve_single_instance, instance_path, H, S, MAX_STEPS)
                    futuros[futuro] = instance_file

                completed_count = 0

                # Recoger resultados de esta carpeta a medida que terminan
                for futuro in as_completed(futuros):
                    instance_file = futuros[futuro]
                    completed_count += 1
                    try:
                        solved, steps, t = futuro.result()

                        # Guardamos en el CSV y forzamos escritura en disco al instante (sin el tiempo)
                        writer.writerow([instance_file, solved, steps])
                        csv_file.flush()

                        print(f"[{completed_count:>2}/{total}] {instance_file:<15} | Solved: {str(solved):<5} | Steps: {steps:<4}")
                    except Exception as exc:
                        print(f"[{completed_count:>2}/{total}] [Error] La instancia {instance_file} falló: {exc}")

    print("\nEvaluación finalizada en todas las carpetas.")


if __name__ == '__main__':
    main()
