import torch
from training.common import load_model
from training.rl import POMOConfig, train
from data.adapters.input import EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1
from models.actions.default import CPMPTransformer
from instances.generators import CVGenerator

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version PyTorch was built with: {torch.version.cuda}")
print(f"Is CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    # Definir los tipos de instancia
    COMBOS = [
        (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        (4, 4), (4, 5), (4, 6), (4, 7),
        (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
        (6, 6), (6, 10)
    ]

    # Configurar el dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() 
                        else "mps" if torch.backends.mps.is_available() 
                        else "cpu")
    print(f"ℹ️ Usando dispositivo: {device}")

    # Cargar el modelo base
    model_name = "actions_sl" 

    model = load_model(CPMPTransformer, model_name)
    model.to(device)
    ref_model = load_model(CPMPTransformer, model_name)
    ref_model.to(device)

    # Inicializar los generadores para cada combo (S, H)
    generators = []
    for S, H in COMBOS:
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
    S_max = 10
    H_max = 8
    input_adapter_config = (EnrichedLayoutAdapter, Layout4DAdapterV1, StackFeaturesAdapterV1, S_max, H_max)

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
        model_name=f"actions_rl",
        test_size_per_combo=640,
        train_size_per_combo=128000,
        seed=42
    )