import os
import glob
from pathlib import Path
from Anomaly_Detection.entity.config_entity import DataIngestionConfig


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
        if not self.config.kaggle_dataset:
            print("No Kaggle dataset configured — skipping download.")
            return

        if self._has_images():
            print(
                f"Dataset already present in {self.config.download_dir} — "
                "skipping download."
            )
            return

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # python-dotenv optional; credentials via kaggle.json also work

        import kaggle
        kaggle.api.authenticate()

        dest = str(self.config.download_dir)
        os.makedirs(dest, exist_ok=True)

        print(f"Downloading '{self.config.kaggle_dataset}' → {dest}")
        kaggle.api.dataset_download_files(
            self.config.kaggle_dataset,
            path=dest,
            unzip=True,
        )
        print(f"Download complete: {dest}")
