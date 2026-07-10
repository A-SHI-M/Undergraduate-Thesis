import os
import sys
import glob
from pathlib import Path
from Anomaly_Detection.entity.config_entity import DataIngestionConfig
from Anomaly_Detection.logger import logger
from Anomaly_Detection.exception import AnomalyDetectionException


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _has_images(self) -> bool:
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            if glob.glob(
                os.path.join(str(self.config.download_dir), "**", ext),
                recursive=True,
            ):
                return True
        return False

    def download(self):
        try:
            if not self.config.kaggle_dataset:
                logger.info("No Kaggle dataset configured — skipping download.")
                return

            if self._has_images():
                logger.info(
                    f"Dataset already present in {self.config.download_dir} — "
                    "skipping download."
                )
                return

            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass

            import kaggle
            kaggle.api.authenticate()

            dest = str(self.config.download_dir)
            os.makedirs(dest, exist_ok=True)

            logger.info(f"Downloading '{self.config.kaggle_dataset}' → {dest}")
            kaggle.api.dataset_download_files(
                self.config.kaggle_dataset,
                path=dest,
                unzip=True,
            )
            logger.info(f"Download complete: {dest}")

        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e
