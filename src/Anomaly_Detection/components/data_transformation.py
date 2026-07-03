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

    # ── image loading ─────────────────────────────────────────────────────────

    def _load_images(self, directory: str) -> np.ndarray:
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))

        images = []
        for path in paths:
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

    # ── Stage 01: transform raw images → save as JPG ────────────────────────────

    def transform(self):
        def _save_images(images: np.ndarray, out_dir: str, label: str):
            os.makedirs(out_dir, exist_ok=True)
            for i, img_arr in enumerate(images):
                img = Image.fromarray((img_arr * 255).astype(np.uint8), mode="L")
                img.save(os.path.join(out_dir, f"{label}_{i:04d}.jpg"))
            print(f"Saved {len(images)} {label} images → {out_dir}")

        normal_images = self._load_images(str(self.config.source_normal_dir))
        abnormal_images = self._load_images(str(self.config.source_abnormal_dir))

        if len(normal_images) == 0:
            print("Source normal images not found — generating dummy data.")
            normal_images = self._create_dummy_normal(self.config.dummy_normal_samples)

        if len(abnormal_images) == 0:
            print("Source abnormal images not found — generating dummy data.")
            abnormal_images = self._create_dummy_abnormal(self.config.dummy_abnormal_samples)

        _save_images(normal_images, str(self.config.normal_dir), "healthy")
        _save_images(abnormal_images, str(self.config.abnormal_dir), "tumor")

    # ── Training: load processed JPGs → return arrays ─────────────────────────

    def initiate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        normal_images = self._load_images(str(self.config.normal_dir))
        abnormal_images = self._load_images(str(self.config.abnormal_dir))

        if len(normal_images) == 0:
            print("Processed images not found — run Stage 01 first, or generating dummy data.")
            normal_images = self._create_dummy_normal(self.config.dummy_normal_samples)

        if len(abnormal_images) == 0:
            abnormal_images = self._create_dummy_abnormal(self.config.dummy_abnormal_samples)

        x_train, x_test_normal = train_test_split(
            normal_images,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        _, x_test_tumor = train_test_split(
            abnormal_images,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        x_test = np.concatenate([x_test_normal, x_test_tumor])
        y_test = np.concatenate([
            np.zeros(len(x_test_normal)),
            np.ones(len(x_test_tumor)),
        ])

        print(f"Train: {x_train.shape}  |  Test: {x_test.shape}  |  Labels: {y_test.shape}")
        return x_train, x_test, y_test
