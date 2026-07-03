import numpy as np
from typing import Dict, Optional, Tuple
from tensorflow.keras.models import load_model, Model
from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.model_evaluation import ModelEvaluator
from Anomaly_Detection.constant import INPUT_DIM, N_CONDITIONS

_MODEL_FILENAMES = {
    "fc_ae": "FC-AE",
    "cnn_ae": "CNN-AE",
    "vae": "VAE",
    "beta_vae": "Beta-VAE",
    "cvae": "CVAE",
    "vqvae": "VQ-VAE",
}


class PredictionPipeline:
    """
    Loads a saved model and scores new images for anomalies.

    Usage:
        pipeline = PredictionPipeline(model_type="bigan", threshold=0.05)
        pipeline.load()
        scores, preds = pipeline.predict(images)
    """

    def __init__(self, model_type: str = "bigan", threshold: Optional[float] = None):
        self.model_type = model_type
        self.threshold = threshold
        eval_cfg = ConfigurationManager().get_model_evaluation_config()
        trainer_cfg = ConfigurationManager().get_model_trainer_config()
        self.models_dir = trainer_cfg.extra_models_dir
        self.evaluator = ModelEvaluator(eval_cfg)
        self._model: Optional[Model] = None
        self._encoder: Optional[Model] = None
        self._generator: Optional[Model] = None

    def load(self):
        def _load(name: str) -> Model:
            path = self.models_dir / f"{name}.keras"
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            return load_model(str(path))

        if self.model_type == "bigan":
            self._encoder = _load("encoder")
            self._generator = _load("generator")
        else:
            self._model = _load(_MODEL_FILENAMES[self.model_type])

    def _preprocess(self, images: np.ndarray) -> np.ndarray:
        if self.model_type in ("fc_ae", "vae", "beta_vae", "cvae"):
            return images.reshape(len(images), INPUT_DIM)
        return images[..., np.newaxis] if images.ndim == 3 else images

    def predict(
        self,
        images: np.ndarray,
        conditions: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (anomaly_scores, binary_predictions). load() must be called first."""
        if self.threshold is None:
            raise ValueError(
                "Set threshold before calling predict(). "
                "Obtain it from ModelEvaluator.evaluate() on a labelled validation set."
            )

        x = self._preprocess(images)

        if self.model_type == "bigan":
            scores = self.evaluator.compute_scores_bigan(self._encoder, self._generator, x)
        elif self.model_type == "cvae":
            if conditions is None:
                conditions = np.zeros((len(x), N_CONDITIONS), dtype=np.float32)
            scores = self.evaluator.compute_scores_cvae(self._model, x, conditions)
        else:
            scores = self.evaluator.compute_scores_autoencoder(self._model, x)

        return scores, (scores >= self.threshold).astype(int)

    def predict_single(self, image: np.ndarray) -> Dict:
        scores, preds = self.predict(image[np.newaxis])
        return {
            "anomaly_score": float(scores[0]),
            "is_anomaly": bool(preds[0]),
            "threshold": self.threshold,
        }
