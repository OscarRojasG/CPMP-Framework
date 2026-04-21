from training.training import load_model, rl_train, Accuracy, DataGenerationConfigRL, save_model
from generation.adapters import EnrichedStackMatrixAdapter, StackMatrix4D3FAdapter, ExtraDataAdapter3F, DefaultMovesAdapter
from models.cpmp_transformer_v5 import CPMPTransformer

model = load_model(CPMPTransformer, "best_v5")

iterations = 5
epochs = 20
train_size = 8000
test_size = 2000
batch_size = 128
learning_rate = 5e-5
weight_decay = 1e-4
patience = 5
metrics = [Accuracy()]
seed = 42

datagen_config = DataGenerationConfigRL(
    instance_set="E5-25-100",
    H=7,
    max_steps=50,
    layout_adapter_config=(EnrichedStackMatrixAdapter, StackMatrix4D3FAdapter, ExtraDataAdapter3F),
    moves_adapter_config=(DefaultMovesAdapter, ),
    num_workers=12
)

model = rl_train(model, iterations, datagen_config, epochs, train_size, test_size, batch_size, learning_rate, weight_decay, patience, metrics, seed)
save_model(model, "rl5")