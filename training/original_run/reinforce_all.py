import torch
from training.sl.training import load_model
from training.rl.training_v2 import POMOConfig, train
from generation.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV2, StackFeaturesAdapterV1
from models.actions.cpmp_transformer_V2 import CPMPTransformer
from instances.generators import CVGenerator, FRGUniformEmpiricalGenerator

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    # Definir los tipos de instancia
    COMBOS = [(3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5),
              (4, 6), (5, 6), (6, 6), (7, 6),
              (4, 7), (5, 7), (6, 7), (7, 7), (8, 7), (9, 7), (10, 7),
              (6, 8), (10, 8)]

    # Configurar el dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available() 
                        else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")

    # Cargar el modelo base
    model_name = "sl_final"

    model = load_model(CPMPTransformer, model_name)
    model.to(device)
    ref_model = load_model(CPMPTransformer, model_name)
    ref_model.to(device)

    # Inicializar los generadores para cada combo (S, H)
    generators = []
    for S, H in COMBOS:
        #gen = FRGUniformEmpiricalGenerator(S=S, H=H, seed=42)
        gen = CVGenerator(S=S, H=H, seed=42)
        generators.append(gen)

    # Configuración principal del entrenamiento
    pomo_config = POMOConfig(
        updates=10000,
        k_rollouts=16,
        instances_per_combo=8,
        kl_coef=0.1,
        adv_clip=4.0,
        grad_clip=1.0,
        minibatch_size=2048,
        eval_interval=10,
        patience=50
    )

    # Adaptador de entrada del modelo
    input_adapter_config = (EnrichedLayoutAdapter, Layout4DAdapterV2, StackFeaturesAdapterV1, 10, 8)

    # Lanzar el entrenamiento
    train(
        model=model,
        ref_model=ref_model,
        generators=generators,
        pomo_config=pomo_config,
        input_adapter_config=input_adapter_config,
        learning_rate=1e-4,
        weight_decay=0.0,
        device=device,
        save_dir="rl",
        model_name=f"reinforce_vf",
        test_size_per_combo=40,
        train_size_per_combo=128000,
        seed=42
    )
