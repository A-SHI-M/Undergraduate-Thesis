import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from tensorflow.keras.models import Model
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from Anomaly_Detection.entity.config_entity import ModelEvaluationConfig
from Anomaly_Detection.constant import IMG_SHAPE, LATENT_DIM
from Anomaly_Detection.logger import logger
from Anomaly_Detection.exception import AnomalyDetectionException


@dataclass
class EvaluationResult:
    auc_roc: float
    average_precision: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    threshold: float
    confusion_mat: np.ndarray
    fpr: np.ndarray
    tpr: np.ndarray
    pr_precision: np.ndarray
    pr_recall: np.ndarray
    anomaly_scores: np.ndarray
    y_test: np.ndarray
    ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    # ── Anomaly score methods ─────────────────────────────────────────────────

    def compute_scores_autoencoder(self, model: Model, x_test: np.ndarray) -> np.ndarray:
        try:
            reconstructed = model.predict(x_test, verbose=0)
            return np.mean(np.square(x_test - reconstructed), axis=tuple(range(1, x_test.ndim)))
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def compute_scores_cvae(
        self, model: Model, x_test: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        reconstructed = model.predict([x_test, conditions], verbose=0)
        return np.mean(np.square(x_test - reconstructed), axis=tuple(range(1, x_test.ndim)))

    def compute_scores_bigan(
        self,
        encoder: Model,
        generator: Model,
        x_test: np.ndarray,
        alpha: float = 0.7,
    ) -> np.ndarray:
        if x_test.ndim == 3:
            x_test = x_test[..., np.newaxis]
        z = encoder.predict(x_test, verbose=0)
        x_rec = generator.predict(z, verbose=0)
        mse_scores = np.mean(np.square(x_test - x_rec), axis=(1, 2, 3))
        ssim_vals = tf.image.ssim(
            tf.convert_to_tensor(x_test, dtype=tf.float32),
            tf.convert_to_tensor(x_rec, dtype=tf.float32),
            max_val=1.0,
        ).numpy()
        return alpha * mse_scores + (1 - alpha) * (1.0 - ssim_vals)

    def compute_scores_ganomaly(
        self,
        encoder1: Model,
        generator: Model,
        encoder2: Model,
        x_test: np.ndarray,
    ) -> np.ndarray:
        if x_test.ndim == 3:
            x_test = x_test[..., np.newaxis]
        z1 = encoder1.predict(x_test, verbose=0)
        x_hat = generator.predict(z1, verbose=0)
        z2 = encoder2.predict(x_hat, verbose=0)
        return np.linalg.norm(z1 - z2, axis=1)

    def compute_scores_fanogan(
        self,
        encoder: Model,
        generator: Model,
        disc_feature_model: Model,
        x_test: np.ndarray,
        kappa: float = 0.1,
    ) -> np.ndarray:
        """f-AnoGAN fast scoring: single forward pass through trained encoder."""
        if x_test.ndim == 3:
            x_test = x_test[..., np.newaxis]
        z_hat = encoder.predict(x_test, verbose=0)
        x_hat = generator.predict(z_hat, verbose=0)
        recon = np.mean(np.square(x_test - x_hat), axis=(1, 2, 3))
        feat_real = disc_feature_model.predict(x_test, verbose=0)
        feat_fake = disc_feature_model.predict(x_hat, verbose=0)
        feat = np.mean(np.square(feat_real - feat_fake), axis=1)
        return (1 - kappa) * recon + kappa * feat

    # ── Confidence intervals ──────────────────────────────────────────────────

    def compute_confidence_intervals(
        self,
        scores: np.ndarray,
        y_test: np.ndarray,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> Dict[str, Tuple[float, float]]:
        rng = np.random.default_rng(42)
        metrics = {k: [] for k in ("auc_roc", "accuracy", "f1", "precision", "recall")}
        fpr_all, tpr_all, thresholds_all = roc_curve(y_test, scores)
        threshold = float(thresholds_all[int(np.argmax(tpr_all - fpr_all))])

        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(y_test), len(y_test))
            s, y = scores[idx], y_test[idx]
            if len(np.unique(y)) < 2:
                continue
            fp, tp, th = roc_curve(y, s)
            metrics["auc_roc"].append(auc(fp, tp))
            y_pred = (s >= threshold).astype(int)
            metrics["accuracy"].append(accuracy_score(y, y_pred))
            metrics["f1"].append(f1_score(y, y_pred, zero_division=0))
            metrics["precision"].append(precision_score(y, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y, y_pred, zero_division=0))

        lo, hi = (1 - ci) / 2, (1 + ci) / 2
        return {k: (float(np.percentile(v, lo * 100)), float(np.percentile(v, hi * 100)))
                for k, v in metrics.items() if v}

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        anomaly_scores: np.ndarray,
        y_test: np.ndarray,
        n_bootstrap: int = 1000,
    ) -> EvaluationResult:
        try:
            fpr, tpr, thresholds = roc_curve(y_test, anomaly_scores)
            roc_auc = auc(fpr, tpr)

            pr_precision, pr_recall, _ = precision_recall_curve(y_test, anomaly_scores)
            ap = average_precision_score(y_test, anomaly_scores)

            best_idx = int(np.argmax(tpr - fpr))
            threshold = float(thresholds[best_idx])

            y_pred = (anomaly_scores >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, _, _ = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

            ci_bounds = self.compute_confidence_intervals(anomaly_scores, y_test, n_bootstrap)

            return EvaluationResult(
                auc_roc=roc_auc,
                average_precision=ap,
                accuracy=accuracy_score(y_test, y_pred),
                precision=precision_score(y_test, y_pred, zero_division=0),
                recall=recall_score(y_test, y_pred, zero_division=0),
                f1=f1_score(y_test, y_pred, zero_division=0),
                specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                threshold=threshold,
                confusion_mat=cm,
                fpr=fpr,
                tpr=tpr,
                pr_precision=pr_precision,
                pr_recall=pr_recall,
                anomaly_scores=anomaly_scores,
                y_test=y_test,
                ci=ci_bounds,
            )
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── Save / print ──────────────────────────────────────────────────────────

    def save_metrics(self, result: EvaluationResult, model_name: str, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "metrics.txt"

        def _ci(key):
            if key in result.ci:
                lo, hi = result.ci[key]
                return f" [{lo:.4f}, {hi:.4f}]"
            return ""

        with open(str(path), "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"AUC-ROC:           {result.auc_roc:.4f}{_ci('auc_roc')}\n")
            f.write(f"Average Precision: {result.average_precision:.4f}\n")
            f.write(f"Accuracy:          {result.accuracy:.4f}{_ci('accuracy')}\n")
            f.write(f"Precision:         {result.precision:.4f}{_ci('precision')}\n")
            f.write(f"Recall:            {result.recall:.4f}{_ci('recall')}\n")
            f.write(f"F1-Score:          {result.f1:.4f}{_ci('f1')}\n")
            f.write(f"Specificity:       {result.specificity:.4f}\n")
            f.write(f"Threshold:         {result.threshold:.6f}\n")
        logger.info(f"Metrics saved → {path}")

    def print_metrics(self, result: EvaluationResult, model_name: str):
        def _ci(key):
            if key in result.ci:
                lo, hi = result.ci[key]
                return f" [{lo:.4f}, {hi:.4f}]"
            return ""

        logger.info(f"── {model_name} ──────────────────────────────")
        logger.info(f"  AUC-ROC:           {result.auc_roc:.4f}{_ci('auc_roc')}")
        logger.info(f"  Average Precision: {result.average_precision:.4f}")
        logger.info(f"  Accuracy:          {result.accuracy:.4f}{_ci('accuracy')}")
        logger.info(f"  Precision:         {result.precision:.4f}{_ci('precision')}")
        logger.info(f"  Recall:            {result.recall:.4f}{_ci('recall')}")
        logger.info(f"  F1-Score:          {result.f1:.4f}{_ci('f1')}")
        logger.info(f"  Specificity:       {result.specificity:.4f}")
