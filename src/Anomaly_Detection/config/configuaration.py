import yaml
from pathlib import Path
from Anomaly_Detection.constant import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Anomaly_Detection.utils import create_directories
from Anomaly_Detection.entity.config_entity import (
    DatasetConfig,
    DataIngestionConfig,
    ModelTrainerConfig,
    BiGANTrainerConfig,
    ModelEvaluationConfig,
    AblationConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        dataset_name: str = None,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        with open(config_filepath) as f:
            self.config = yaml.safe_load(f)
        with open(params_filepath) as f:
            self.params = yaml.safe_load(f)

        # Resolve active dataset
        datasets = self.config["datasets"]
        ds_key = dataset_name or list(datasets.keys())[0]
        if ds_key not in datasets:
            raise ValueError(f"Dataset '{ds_key}' not found in config.yaml. "
                             f"Available: {list(datasets.keys())}")
        ds = datasets[ds_key]
        self.dataset_name = ds_key
        self.display_name = ds["display_name"]

        create_directories([self.config["artifacts_root"]])

    def _resolve(self, template: str) -> str:
        return template.replace("{dataset}", self.display_name)

    def get_dataset_config(self) -> DatasetConfig:
        ds = self.config["datasets"][self.dataset_name]
        return DatasetConfig(
            name=self.dataset_name,
            display_name=self.display_name,
            source_normal_dir=Path(ds["source_normal_dir"]),
            source_abnormal_dir=Path(ds["source_abnormal_dir"]),
        )

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        ds = self.config["datasets"][self.dataset_name]
        p = self.params["data"]

        root = self._resolve(cfg["root_dir"])
        normal = self._resolve(cfg["normal_dir"])
        abnormal = self._resolve(cfg["abnormal_dir"])
        create_directories([root, normal, abnormal])

        return DataIngestionConfig(
            root_dir=Path(root),
            source_normal_dir=Path(ds["source_normal_dir"]),
            source_abnormal_dir=Path(ds["source_abnormal_dir"]),
            normal_dir=Path(normal),
            abnormal_dir=Path(abnormal),
            img_size=tuple(p["img_size"]),
            test_size=float(p["test_size"]),
            random_state=int(p["random_state"]),
            dummy_normal_samples=int(p["dummy_normal_samples"]),
            dummy_abnormal_samples=int(p["dummy_abnormal_samples"]),
        )

    def get_model_trainer_config(self, model_name: str) -> ModelTrainerConfig:
        cfg = self.config["model_trainer"]
        p = self.params[model_name]

        root = self._resolve(cfg["root_dir"])
        models = self._resolve(cfg["models_dir"])
        extra = self._resolve(cfg["extra_models_dir"])
        create_directories([root, models, extra])

        return ModelTrainerConfig(
            root_dir=Path(root),
            models_dir=Path(models),
            extra_models_dir=Path(extra),
            epochs=int(p["epochs"]),
            batch_size=int(p["batch_size"]),
            validation_split=float(p["validation_split"]),
        )

    def get_bigan_trainer_config(self) -> BiGANTrainerConfig:
        cfg = self.config["model_trainer"]
        p = self.params["bigan"]

        root = self._resolve(cfg["root_dir"])
        models = self._resolve(cfg["models_dir"])
        extra = self._resolve(cfg["extra_models_dir"])
        create_directories([root, models, extra])

        return BiGANTrainerConfig(
            root_dir=Path(root),
            models_dir=Path(models),
            extra_models_dir=Path(extra),
            pretrain_epochs=int(p["pretrain_epochs"]),
            bigan_epochs=int(p["bigan_epochs"]),
            disc_train_interval=int(p["disc_train_interval"]),
            learning_rate=float(p["learning_rate"]),
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        cfg = self.config["model_evaluation"]
        root = self._resolve(cfg["root_dir"])
        comparison = self._resolve(cfg["comparison_dir"])
        create_directories([root, comparison])
        return ModelEvaluationConfig(
            root_dir=Path(root),
            comparison_dir=Path(comparison),
        )

    def get_ablation_config(self) -> AblationConfig:
        cfg = self.config["ablation_analysis"]
        root = self._resolve(cfg["root_dir"])
        comparison = self._resolve(cfg["comparison_dir"])
        create_directories([root, comparison])
        return AblationConfig(
            root_dir=Path(root),
            comparison_dir=Path(comparison),
        )
