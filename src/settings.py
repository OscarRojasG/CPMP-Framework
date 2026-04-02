from pathlib import Path

PROJECT_ROOT = Path.cwd().parent
INSTANCE_FOLDER = PROJECT_ROOT / "instances"
DATA_FOLDER = PROJECT_ROOT / "data"
MODELS_FOLDER = PROJECT_ROOT / "models"
HYPERPARAMETERS_FOLDER = MODELS_FOLDER / "hyperparameters"

FEG_PATH = PROJECT_ROOT / "src" / "feg"