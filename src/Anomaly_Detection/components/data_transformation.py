import os
import glob
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Tuple
from Anomaly_Detection.entity.config_entity import DataIngestionConfig


class DataTransformation:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _load_images(self, directory: str) -> np.ndarray:
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))

        images = []
        for path in sorted(paths):
            try:
                img = Image.open(path).convert("L").resize(self.config.img_size)
                images.append(np.array(img, dtype=np.float32) / 255.0)
            except Exception:
                continue
        return np.array(images) if images else np.array([])

    def _create_dummy_normal(self, n: int) -> np.ndarray:
        h, w = self.config.img_size
        images = []
        for _ in range(n):
            img = np.zeros((h, w), dtype=np.float32)
            cx, cy = np.random.randint(30, h - 30), np.random.randint(30, w - 30)
            Y, X = np.ogrid[:h, :w]
            mask = (X - cx) ** 2 + (Y - cy) ** 2 <= 900
            img[mask] = np.random.uniform(0.5, 1.0)
            img += np.random.normal(0, 0.05, img.shape)
            images.append(np.clip(img, 0, 1))
        return np.array(images)

    def _create_dummy_abnormal(self, n: int) -> np.ndarray:
        h, w = self.config.img_size
        normals = self._create_dummy_normal(n)
        for img in normals:
            rx, ry = np.random.randint(20, h - 40), np.random.randint(20, w - 40)
            size = np.random.randint(10, 30)
            img[rx: rx + size, ry: ry + size] = 1.0
        return normals

    def initiate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load images from Datasets/{dataset}/normal and abnormal, return train/test splits."""
        normal_images   = self._load_images(str(self.config.normal_dir))
        abnormal_images = self._load_images(str(self.config.abnormal_dir))

        if len(normal_images) == 0:
            print(
                f"No images found in {self.config.normal_dir} — "
                "generating dummy normal data."
            )
            normal_images = self._create_dummy_normal(self.config.dummy_normal_samples)

        if len(abnormal_images) == 0:
            print(
                f"No images found in {self.config.abnormal_dir} — "
                "generating dummy abnormal data."
            )
            abnormal_images = self._create_dummy_abnormal(self.config.dummy_abnormal_samples)

        x_train, x_test_normal = train_test_split(
            normal_images,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        _, x_test_abnormal = train_test_split(
            abnormal_images,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        x_test = np.concatenate([x_test_normal, x_test_abnormal])
        y_test = np.concatenate([
            np.zeros(len(x_test_normal)),
            np.ones(len(x_test_abnormal)),
        ])

        print(f"Train: {x_train.shape}  |  Test: {x_test.shape}  |  Labels: {y_test.shape}")
        return x_train, x_test, y_test
