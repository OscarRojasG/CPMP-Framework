from models.actions.v1 import CPMPTransformer
from models.cost.v1 import CostPredictorTransformer
from training.common import load_model
from data.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1
from solvers import BSGCostPredictorSolver
from evaluators.bs_evaluator import run_eval

def run_all_benchmarks():
    # 1. Cargamos el modelo una sola vez (reutilizamos los pesos para todas las configs)
    action_model_name = "rl"
    action_model = load_model(CPMPTransformer, action_model_name)

    cost_model_name = "cost"
    cost_model = load_model(CostPredictorTransformer, cost_model_name)
    
    # 2. Definimos las configuraciones y los anchos de Beam Search (w)
    w_values = [2, 4, 8, 16, 32]
    configs = [
        (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        (4, 4), (4, 5), (4, 6), (4, 7),
        (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 6), (6, 10)
    ]
    
    max_steps = 100

    print("=== INICIANDO BATERÍA DE BENCHMARKS ===")
    
    # 3. Iteramos por cada configuración de entorno
    for H_minus_2, S in configs:
        # Recuperamos el valor real de H eliminando el offset
        H = H_minus_2 + 2
        folder = f"benchmarks/{H_minus_2}-{S}"
        
        print(f"\n{'='*50}")
        print(f"Configuración Actual -> S: {S} | H: {H} (Carpeta: {folder})")
        print(f"{'='*50}")
        
        # 4. Recreamos el input_adapter para las dimensiones de esta configuración
        input_adapter = EnrichedLayoutAdapter(
            Layout4DAdapterV1, 
            StackFeaturesAdapterV1, 
            S, 
            H
        )
        
        # 5. Ejecutamos la evaluación para cada valor de w
        for w in w_values:
            print(f"\n--- Evaluando con w={w} ---")
            
            # Instanciamos el solver con el w y el adaptador actualizados
            solver = BSGCostPredictorSolver(action_model, cost_model, input_adapter, w)
            
            # Llamamos a tu función refactorizada
            run_eval(solver, folder, H, max_steps)

if __name__ == "__main__":
    run_all_benchmarks()