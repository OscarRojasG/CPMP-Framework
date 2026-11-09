from instances import generate_instances
from instances.generators import FRGUniformEmpiricalGenerator
from data.generation import generate_data_solver
from data.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1
from data.adapters.output import ActionAdapter
from solvers import FRGSolver

if __name__ == "__main__":
    # Configuraciones de instancias a utilizar
    configs = [(3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
               (4, 4), (4, 5), (4, 6), (4, 7),
               (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
               (6, 6), (6, 10)]

    # Generamos instancias de cada tipo
    for H, S in configs:
        ue_generator = FRGUniformEmpiricalGenerator(H=H+2, S=S, seed=42)
        generate_instances(basename=f"ue_sl_{H}-{S}", generator=ue_generator, amount=20000)

    # Configuración de la generación
    max_steps = 100
    num_workers = None
    output_prefix = None
    input_adapter_config = (EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1, 10, 8)
    output_adapter_config = (ActionAdapter, )
    solver_config = (FRGSolver, )

    # Generamos los datos etiquetando cada instancia con las acciones óptimas según FRG
    for H, S in configs:
        folder = f"ue_sl_{H}-{S}"
        generate_data_solver(
            folder,
            H+2,
            max_steps,
            input_adapter_config,
            output_adapter_config,
            solver_config,
            num_workers,
            output_prefix
        )
