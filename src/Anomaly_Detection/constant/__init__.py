from pathlib import Path

CONFIG_FILE_PATH = Path("config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

IMG_SIZE = (128, 128)
IMG_HEIGHT, IMG_WIDTH = IMG_SIZE
INPUT_DIM = IMG_HEIGHT * IMG_WIDTH   # 16384
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)

LATENT_DIM = 128
N_CONDITIONS = 3
BETA_VALUE = 10.0

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.00005

ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "artifacts/model_trainer/saved_models"
PLOTS_DIR = "artifacts/model_evaluation/plots"
METRICS_DIR = "artifacts/model_evaluation/metrics"
