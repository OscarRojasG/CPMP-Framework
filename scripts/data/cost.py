from data.generation import generate_data_model
from data.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1
from data.adapters.output import CostAdapter
from training.common import load_model
from models.actions.default import CPMPTransformer

if __name__ == "__main__":
    # Configuraciones de instancias a utilizar
    configs = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
               (4, 4), (4, 5), (4, 6), (4, 7),
               (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
               (6, 6), (6, 10)]

    files = [(f"ue_sl_{H}-{S}", H, S) for H, S in configs]

    # Configuración del proceso
    max_steps = 100
    input_adapter_config = (EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1, 10, 8)
    output_adapter_config = (CostAdapter, )
    model = load_model(CPMPTransformer, "rl")
    batch_size = 32
    num_workers = None

    # Ejecutamos la generación de datos a partir de las trayectorias del modelo
    for file, hx, sx in files:
        output_name = f"cost_{hx}-{sx}.data"
        generate_data_model(
            file,
            hx+2,
            max_steps,
            input_adapter_config,
            output_adapter_config,
            model,
            batch_size,
            num_workers,
            output_name,
            verbose=True
        )
