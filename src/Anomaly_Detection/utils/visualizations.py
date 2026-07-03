import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from Anomaly_Detection.components.model_evaluation import EvaluationResult
from Anomaly_Detection.constant import IMG_SIZE


class Visualizer:
    @staticmethod
    def _save(path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(path), bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved plot → {path}")

    # ── Per-model plots ───────────────────────────────────────────────────────

    def plot_roc_curve(self, result: EvaluationResult, model_name: str, output_dir: Path):
        plt.figure(figsize=(6, 5))
        plt.plot(result.fpr, result.tpr, label=f"AUC = {result.auc_roc:.3f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {model_name}")
        plt.legend()
        self._save(Path(output_dir) / "roc_curve.png")

    def plot_pr_curve(self, result: EvaluationResult, model_name: str, output_dir: Path):
        plt.figure(figsize=(6, 5))
        plt.plot(result.pr_recall, result.pr_precision, label=f"AP = {result.average_precision:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall — {model_name}")
        plt.legend()
        self._save(Path(output_dir) / "pr_curve.png")

    def plot_confusion_matrix(self, result: EvaluationResult, model_name: str, output_dir: Path):
        cm = result.confusion_mat
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap='Blues')
        plt.colorbar(im, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        self._save(Path(output_dir) / "confusion_matrix.png")

    def plot_score_distribution(self, result: EvaluationResult, model_name: str, output_dir: Path):
        normal_scores = result.anomaly_scores[result.y_test == 0]
        anomaly_scores = result.anomaly_scores[result.y_test == 1]
        plt.figure(figsize=(7, 4))
        plt.hist(normal_scores, bins=40, alpha=0.6, label='Normal', color='blue')
        plt.hist(anomaly_scores, bins=40, alpha=0.6, label='Anomaly', color='red')
        plt.axvline(result.threshold, color='black', linestyle='--',
                    label=f'Threshold={result.threshold:.4f}')
        plt.xlabel("Anomaly Score")
        plt.ylabel("Count")
        plt.title(f"Score Distribution — {model_name}")
        plt.legend()
        self._save(Path(output_dir) / "score_distribution.png")

    def plot_heatmap(
        self,
        model,
        x_samples: np.ndarray,
        model_name: str,
        output_dir: Path,
        n: int = 5,
        label: str = "sample",
        cvae_conditions: Optional[np.ndarray] = None,
    ):
        n = min(n, len(x_samples))
        x_input = x_samples[:n]

        if cvae_conditions is not None:
            reconstructed = model.predict([x_input, cvae_conditions[:n]], verbose=0)
        else:
            reconstructed = model.predict(x_input, verbose=0)

        h, w = IMG_SIZE
        fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 7))
        for i in range(n):
            orig = x_input[i].reshape(h, w)
            recon = reconstructed[i].reshape(h, w)
            error = np.abs(orig - recon)

            axes[0, i].imshow(orig, cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            axes[1, i].imshow(recon, cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            axes[2, i].imshow(orig, cmap='gray', vmin=0, vmax=1)
            axes[2, i].imshow(error, cmap='hot', alpha=0.6, vmin=0, vmax=max(error.max(), 1e-6))
            axes[2, i].axis('off')

        axes[0, 0].set_ylabel("Original", fontsize=9)
        axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
        axes[2, 0].set_ylabel("Heatmap", fontsize=9)
        plt.suptitle(f"{model_name} — {label.capitalize()} Heatmap")
        plt.tight_layout()
        self._save(Path(output_dir) / f"heatmap_{label}.png")

    # ── T-SNE ─────────────────────────────────────────────────────────────────

    def plot_tsne(
        self,
        latent: np.ndarray,
        y_test: np.ndarray,
        model_name: str,
        output_dir: Path,
    ):
        from sklearn.manifold import TSNE
        flat = latent.reshape(len(latent), -1)
        embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(flat)
        plt.figure(figsize=(6, 5))
        plt.scatter(embedded[y_test == 0, 0], embedded[y_test == 0, 1],
                    c='blue', alpha=0.5, s=10, label='Normal')
        plt.scatter(embedded[y_test == 1, 0], embedded[y_test == 1, 1],
                    c='red', alpha=0.5, s=10, label='Anomaly')
        plt.title(f"T-SNE Latent Space — {model_name}")
        plt.legend()
        self._save(Path(output_dir) / "tsne.png")

    def plot_tsne_comparison(
        self,
        latent_dict: Dict[str, np.ndarray],
        y_test: np.ndarray,
        comparison_dir: Path,
    ):
        from sklearn.manifold import TSNE
        n = len(latent_dict)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 10))
        axes = np.array(axes).flatten()
        for ax, (name, latent) in zip(axes, latent_dict.items()):
            flat = latent.reshape(len(latent), -1)
            embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(flat)
            ax.scatter(embedded[y_test == 0, 0], embedded[y_test == 0, 1],
                       c='blue', alpha=0.5, s=8, label='Normal')
            ax.scatter(embedded[y_test == 1, 0], embedded[y_test == 1, 1],
                       c='red', alpha=0.5, s=8, label='Anomaly')
            ax.set_title(name, fontsize=10)
            ax.legend(fontsize=7)
        for ax in axes[n:]:
            ax.set_visible(False)
        plt.suptitle("T-SNE Latent Space Comparison", fontsize=13)
        plt.tight_layout()
        self._save(Path(comparison_dir) / "tsne_comparison.png")

    # ── Comparison plots ──────────────────────────────────────────────────────

    def plot_comparison(
        self,
        results: Dict[str, EvaluationResult],
        comparison_dir: Path,
        metrics: List[str] = ("auc_roc", "accuracy", "precision", "recall", "f1", "specificity"),
    ):
        model_names = list(results.keys())
        x = np.arange(len(model_names))
        width = 0.12
        fig, ax = plt.subplots(figsize=(14, 5))
        for i, metric in enumerate(metrics):
            values = [getattr(results[m], metric) for m in model_names]
            ax.bar(x + i * width, values, width, label=metric.replace("_", " ").title())
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(model_names, rotation=20, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — All Metrics")
        ax.legend(fontsize=7)
        plt.tight_layout()
        self._save(Path(comparison_dir) / "metrics_comparison.png")

    def plot_roc_comparison(
        self,
        results: Dict[str, EvaluationResult],
        comparison_dir: Path,
    ):
        plt.figure(figsize=(8, 6))
        for name, result in results.items():
            plt.plot(result.fpr, result.tpr, label=f"{name} (AUC={result.auc_roc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve Comparison")
        plt.legend(fontsize=7)
        self._save(Path(comparison_dir) / "roc_comparison.png")
