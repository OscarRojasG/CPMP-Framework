from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
WORK_PATH = Path(os.getenv("WORK_PATH"))

INSTANCE_FOLDER = WORK_PATH
DATA_FOLDER = WORK_PATH
MODELS_FOLDER = WORK_PATH
HYPERPARAMETERS_FOLDER = WORK_PATH

PROJECT_ROOT = Path(__file__).parent.parent
FRG_PATH = PROJECT_ROOT / "src" / "frg"