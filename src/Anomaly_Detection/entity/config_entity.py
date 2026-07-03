from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    display_name: str
    source_normal_dir: Path
    source_abnormal_dir: Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_normal_dir: Path
    source_abnormal_dir: Path
    normal_dir: Path
    abnormal_dir: Path
    img_size: Tuple[int, int]
    test_size: float
    random_state: int
    dummy_normal_samples: int
    dummy_abnormal_samples: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    models_dir: Path
    extra_models_dir: Path
    epochs: int
    batch_size: int
    validation_split: float


@dataclass(frozen=True)
class BiGANTrainerConfig:
    root_dir: Path
    models_dir: Path
    extra_models_dir: Path
    pretrain_epochs: int
    bigan_epochs: int
    disc_train_interval: int
    learning_rate: float


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    comparison_dir: Path


@dataclass(frozen=True)
class AblationConfig:
    root_dir: Path
    comparison_dir: Path
