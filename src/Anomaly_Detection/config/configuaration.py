import yaml
from pathlib import Path
from Anomaly_Detection.constant import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from Anomaly_Detection.utils import create_directories
from Anomaly_Detection.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
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

        # Dataset registry lives inside data_ingestion.datasets
        datasets = self.config["data_ingestion"]["datasets"]
        ds_key = dataset_name or list(datasets.keys())[0]
        if ds_key not in datasets:
            raise ValueError(
                f"Dataset '{ds_key}' not found in config.yaml. "
                f"Available: {list(datasets.keys())}"
            )
        ds = datasets[ds_key]
        self.dataset_name = ds_key
        self.display_name = ds["display_name"]

        create_directories([self.config["artifacts_root"]])

    def _resolve(self, template: str) -> str:
        return template.replace("{dataset}", self.display_name)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        ds  = cfg["datasets"][self.dataset_name]
        download_dir = Path(self._resolve(cfg["download_dir"]))
        create_directories([download_dir])
        return DataIngestionConfig(
            download_dir=download_dir,
            kaggle_dataset=ds.get("kaggle_dataset", ""),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        di_cfg = self.config["data_ingestion"]
        dt_cfg = self.config["data_transformation"]
        dt_ds  = dt_cfg["datasets"][self.dataset_name]
        p      = self.params["data"]

        download_dir = Path(self._resolve(di_cfg["download_dir"]))
        root_dir     = Path(self._resolve(dt_cfg["root_dir"]))
        normal_dir   = download_dir / dt_ds["normal_subdir"]
        abnormal_dir = download_dir / dt_ds["abnormal_subdir"]
        create_directories([root_dir])

        return DataTransformationConfig(
            root_dir=root_dir,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            img_size=tuple(p["img_size"]),
            test_size=float(p["test_size"]),
            random_state=int(p["random_state"]),
            target_normal=int(p["target_normal"]),
            target_abnormal=int(p["target_abnormal"]),
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
