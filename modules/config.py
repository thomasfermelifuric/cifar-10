from pathlib import Path

# Get the directory of the current file
current_dir = Path(__file__).parent

# Construct the path to the root directory
root_dir = current_dir.parent

# Construct the path to the data directory
DATA_DIR = root_dir / 'data' / 'cifar-10-batches-py'

# Construct the path to the models directory
MODELS_DIR = root_dir / 'models'

N_TRAIN_SAMPLES = 50000
N_TEST_SAMPLES = 10000
