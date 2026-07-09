import os
import glob
import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import List, Tuple
from pathlib import Path
from Anomaly_Detection.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _collect_paths(self, directory: str) -> List[str]:
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
        return sorted(paths)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).copy()

    def _augment(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        # Flip
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        # Translation (up to 10% of image dimensions)
        tx = int(random.uniform(-0.1, 0.1) * w)
        ty = int(random.uniform(-0.1, 0.1) * h)
        img = img.transform(img.size, Image.AFFINE, (1, 0, -tx, 0, 1, -ty))
        # Scaling (0.8–1.2), crop or pad back to original size
        scale = random.uniform(0.8, 1.2)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        if scale > 1.0:
            left = (new_w - w) // 2
            top  = (new_h - h) // 2
            img  = img.crop((left, top, left + w, top + h))
        else:
            canvas = Image.new(img.mode, (w, h), 0)
            canvas.paste(img, ((w - new_w) // 2, (h - new_h) // 2))
            img = canvas
        return img

    def _preprocess(self, img: Image.Image) -> np.ndarray:
        img = img.convert("L").resize(self.config.img_size)
        return np.array(img, dtype=np.float32) / 255.0

    def _reach_target(self, paths: List[str], target: int) -> List[np.ndarray]:
        rng = random.Random(self.config.random_state)
        if len(paths) >= target:
            selected = rng.sample(paths, target)
            return [self._preprocess(self._load_image(p)) for p in selected]
        # Pre-resize to img_size so augmentation runs on small images
        small: List[Image.Image] = []
        for p in paths:
            try:
                small.append(self._load_image(p).convert("L").resize(self.config.img_size))
            except Exception:
                continue
        result = [np.array(img, dtype=np.float32) / 255.0 for img in small]
        while len(result) < target:
            aug = self._augment(rng.choice(small))
            result.append(np.array(aug, dtype=np.float32) / 255.0)
        return result

    def initiate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        root = Path(self.config.root_dir)
        x_train_path = root / "x_train.npy"

        if x_train_path.exists():
            print(f"Loading cached data from {root}")
            x_train = np.load(root / "x_train.npy")
            x_test  = np.load(root / "x_test.npy")
            y_test  = np.load(root / "y_test.npy")
            print(f"Train: {x_train.shape}  |  Test: {x_test.shape}  |  Labels: {y_test.shape}")
            return x_train, x_test, y_test

        normal_paths   = self._collect_paths(str(self.config.normal_dir))
        abnormal_paths = self._collect_paths(str(self.config.abnormal_dir))

        if len(normal_paths) == 0:
            print(f"No images in {self.config.normal_dir} — using dummy data.")
            h, w = self.config.img_size
            normal_arr   = np.random.rand(self.config.dummy_normal_samples,   h, w).astype(np.float32)
            abnormal_arr = np.random.rand(self.config.dummy_abnormal_samples, h, w).astype(np.float32)
        else:
            print(f"Normal: {len(normal_paths)} images  |  Abnormal: {len(abnormal_paths)} images")
            normal_arr   = np.array(self._reach_target(normal_paths,   self.config.target_normal))
            abnormal_arr = np.array(self._reach_target(abnormal_paths, self.config.target_abnormal))

        # 80% normal → train, 20% normal + all abnormal → test
        x_train, x_test_normal = train_test_split(
            normal_arr,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        x_test = np.concatenate([x_test_normal, abnormal_arr])
        y_test = np.concatenate([
            np.zeros(len(x_test_normal), dtype=np.float32),
            np.ones(len(abnormal_arr),   dtype=np.float32),
        ])

        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "x_train.npy", x_train)
        np.save(root / "x_test.npy",  x_test)
        np.save(root / "y_test.npy",  y_test)
        print(f"Saved to {root}")
        print(f"Train: {x_train.shape}  |  Test: {x_test.shape}  |  Labels: {y_test.shape}")
        return x_train, x_test, y_test
