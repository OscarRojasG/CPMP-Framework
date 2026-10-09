# ==================== Parámetros ====================
# True: modelo con memoria (default.py) | False: modelo sin memoria (no_memory.py)
USE_MEMORY = True

# Conjunto de benchmarks a evaluar: "cvs", "g2" o "g3"
DATASET = "cvs"

# Modelo a cargar (nombre en la carpeta models/)
MODEL_NAME = "actions_rl"

# Anchos de Beam Search (w)
W_VALUES = [2, 4, 8, 16, 32]

# Configuraciones (H-2, S)
CONFIGS = [
    (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
    (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
    (6, 6), (6, 10)
]

MAX_STEPS = 100
# ====================================================

import os
if USE_MEMORY:
    from models.actions.default import CPMPTransformer
else:
    from models.actions.no_memory import CPMPTransformer
from training.common import load_model
from data.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1
from solvers import BSGModelSolver
from evaluators.evaluator import run_eval

def run_all_benchmarks():
    # 1. Cargamos el modelo una sola vez (reutilizamos los pesos para todas las configs)
    model = load_model(CPMPTransformer, MODEL_NAME)

    print("=== INICIANDO BATERÍA DE BENCHMARKS ===")

    # 2. Iteramos por cada configuración de entorno
    for H_minus_2, S in CONFIGS:
        # Recuperamos el valor real de H eliminando el offset
        H = H_minus_2 + 2
        folder = f"benchmarks/{DATASET}/{H_minus_2}-{S}"

        print(f"\n{'='*50}")
        print(f"Configuración Actual -> S: {S} | H: {H} (Carpeta: {folder})")
        print(f"{'='*50}")

        # 3. Recreamos el input_adapter para las dimensiones de esta configuración
        input_adapter = EnrichedLayoutAdapter(
            Layout4DAdapterV1,
            StackFeaturesAdapterV1,
            S,
            H
        )

        # 4. Ejecutamos la evaluación para cada valor de w
        for w in W_VALUES:
            print(f"\n--- Evaluando con w={w} ---")

            # Instanciamos el solver con el w y el adaptador actualizados
            solver = BSGModelSolver(model, input_adapter, w)

            # Los resultados sin memoria llevan el prefijo "nm_" para no sobrescribir los con memoria
            prefix = "" if USE_MEMORY else "nm_"
            csv_filename = f"{prefix}{solver.name}_w{w}_{DATASET}_{H_minus_2}-{S}.csv"

            # Llamamos a tu función refactorizada
            run_eval(solver, folder, H, MAX_STEPS, csv_filename)

if __name__ == "__main__":
    run_all_benchmarks()
