import sys
import mlflow
import numpy as np
import tensorflow as tf
from pathlib import Path
from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.model_evaluation import ModelEvaluator, EvaluationResult
from Anomaly_Detection.utils.visualizations import Visualizer
from Anomaly_Detection.utils.mlflow_tracker import setup_mlflow
from Anomaly_Detection.constant import N_CONDITIONS
from Anomaly_Detection.logger import logger
from Anomaly_Detection.exception import AnomalyDetectionException


class _ReconWrapper:
    """Wraps encoder + generator so GAN models behave like autoencoders for heatmap."""
    def __init__(self, encoder, generator):
        self._enc = encoder
        self._gen = generator

    def predict(self, x, verbose=0):
        return self._gen.predict(self._enc.predict(x, verbose=verbose), verbose=verbose)


class EvaluationPipeline:
    def __init__(self, dataset_name: str = None):
        try:
            self.cfg = ConfigurationManager(dataset_name=dataset_name)
            self._eval_cfg = self.cfg.get_model_evaluation_config()
            self._data_cfg = self.cfg.get_data_transformation_config()
            _, self._models_dir, _ = self.cfg.get_model_dirs()
            self._ablation_cfg = self.cfg.get_ablation_config()
            self._ev = ModelEvaluator(self._eval_cfg)
            self._viz = Visualizer()
            experiment_name = self.cfg.config["mlflow"]["experiment_name"]
            setup_mlflow(experiment_name)
            logger.info(f"EvaluationPipeline initialized for dataset: {dataset_name}")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── Data ──────────────────────────────────────────────────────────────────

    def _load_test_data(self):
        try:
            root = Path(self._data_cfg.root_dir)
            x_test = np.load(root / "x_test.npy")
            y_test = np.load(root / "y_test.npy")
            logger.info(f"Test: {x_test.shape}  |  Labels: {y_test.shape}")
            return x_test, y_test
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _try_load(self, *names):
        models = []
        for name in names:
            p = self._models_dir / name
            if not p.exists():
                logger.warning(f"Skipping — model not found: {p}")
                return None
            try:
                models.append(tf.keras.models.load_model(str(p)))
            except Exception as e:
                logger.warning(f"Skipping — failed to load {p}: {e}")
                return None
        return models

    def _out_dir(self, name: str) -> Path:
        d = Path(self._eval_cfg.root_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _full_eval(
        self,
        scores: np.ndarray,
        y_test: np.ndarray,
        name: str,
        params_key: str,
        heatmap_model=None,
        heatmap_x: np.ndarray = None,
        cvae_cond: np.ndarray = None,
        out_dir: Path = None,
    ) -> EvaluationResult:
        try:
            out = out_dir if out_dir is not None else self._out_dir(name)
            out.mkdir(parents=True, exist_ok=True)
            result = self._ev.evaluate(scores, y_test)
            self._ev.print_metrics(result, name)
            self._ev.save_metrics(result, name, out)
            self._viz.plot_roc_curve(result, name, out)
            self._viz.plot_pr_curve(result, name, out)
            self._viz.plot_confusion_matrix(result, name, out)
            self._viz.plot_score_distribution(result, name, out)
            if heatmap_model is not None and heatmap_x is not None:
                for label, mask in (("normal", y_test == 0), ("anomaly", y_test == 1)):
                    if mask.any():
                        cond = cvae_cond[mask] if cvae_cond is not None else None
                        self._viz.plot_heatmap(
                            heatmap_model, heatmap_x[mask], name, out,
                            label=label, cvae_conditions=cond,
                        )
            params = self.cfg.params.get(params_key, {})
            with mlflow.start_run(run_name=name):
                if params:
                    mlflow.log_params(params)
                mlflow.log_metrics({
                    "auc_roc":           result.auc_roc,
                    "average_precision": result.average_precision,
                    "accuracy":          result.accuracy,
                    "precision":         result.precision,
                    "recall":            result.recall,
                    "f1":                result.f1,
                    "specificity":       result.specificity,
                })
            logger.info(f"Evaluation complete for {name} — AUC-ROC: {result.auc_roc:.4f}")
            return result
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── Per-model evaluation ──────────────────────────────────────────────────

    def eval_fc_ae(self, x_test, y_test):
        try:
            m = self._try_load("FC-AE.keras")
            if m is None:
                return None
            x = x_test.reshape(x_test.shape[0], -1)
            scores = self._ev.compute_scores_autoencoder(m[0], x)
            return self._full_eval(scores, y_test, "FC-AE", "fc_ae",
                                   heatmap_model=m[0], heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_cnn_ae(self, x_test, y_test):
        try:
            m = self._try_load("CNN-AE.keras")
            if m is None:
                return None
            x = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
            scores = self._ev.compute_scores_autoencoder(m[0], x)
            return self._full_eval(scores, y_test, "CNN-AE", "cnn_ae",
                                   heatmap_model=m[0], heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_vae(self, x_test, y_test):
        try:
            m = self._try_load("VAE.keras")
            if m is None:
                return None
            x = x_test.reshape(x_test.shape[0], -1)
            scores = self._ev.compute_scores_autoencoder(m[0], x)
            return self._full_eval(scores, y_test, "VAE", "vae",
                                   heatmap_model=m[0], heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_beta_vae(self, x_test, y_test):
        try:
            m = self._try_load("Beta-VAE.keras")
            if m is None:
                return None
            x = x_test.reshape(x_test.shape[0], -1)
            scores = self._ev.compute_scores_autoencoder(m[0], x)
            return self._full_eval(scores, y_test, "Beta-VAE", "beta_vae",
                                   heatmap_model=m[0], heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_cvae(self, x_test, y_test):
        try:
            m = self._try_load("CVAE.keras")
            if m is None:
                return None
            x = x_test.reshape(x_test.shape[0], -1)
            cond = np.zeros((len(x), N_CONDITIONS), dtype=np.float32)
            scores = self._ev.compute_scores_cvae(m[0], x, cond)
            return self._full_eval(scores, y_test, "CVAE", "cvae",
                                   heatmap_model=m[0], heatmap_x=x, cvae_cond=cond)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_vqvae(self, x_test, y_test):
        try:
            m = self._try_load("VQ-VAE.keras")
            if m is None:
                return None
            x = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
            scores = self._ev.compute_scores_autoencoder(m[0], x)
            return self._full_eval(scores, y_test, "VQ-VAE", "vqvae",
                                   heatmap_model=m[0], heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_normal_bigan(self, x_test, y_test):
        try:
            m = self._try_load("bigan_encoder.keras", "bigan_generator.keras")
            if m is None:
                return None
            encoder, generator = m
            x = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
            scores = self._ev.compute_scores_bigan(encoder, generator, x)
            return self._full_eval(scores, y_test, "BiGAN", "bigan_normal",
                                   heatmap_model=_ReconWrapper(encoder, generator), heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_bigan(self, x_test, y_test):
        try:
            m = self._try_load("improved_bigan_encoder.keras", "improved_bigan_generator.keras")
            if m is None:
                return None
            encoder, generator = m
            x = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
            scores = self._ev.compute_scores_bigan(encoder, generator, x)
            return self._full_eval(scores, y_test, "Improved-BiGAN", "bigan",
                                   heatmap_model=_ReconWrapper(encoder, generator), heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_ganomaly(self, x_test, y_test):
        try:
            m = self._try_load("ganomaly_encoder1.keras", "ganomaly_generator.keras",
                               "ganomaly_encoder2.keras")
            if m is None:
                return None
            encoder1, generator, encoder2 = m
            x = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
            scores = self._ev.compute_scores_ganomaly(encoder1, generator, encoder2, x)
            return self._full_eval(scores, y_test, "GANomaly", "ganomaly",
                                   heatmap_model=_ReconWrapper(encoder1, generator), heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def eval_fanogan(self, x_test, y_test):
        try:
            m = self._try_load("fanogan_encoder.keras", "fanogan_generator.keras",
                               "fanogan_discriminator.keras")
            if m is None:
                return None
            encoder, generator, discriminator = m
            x = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
            feat_layer = next(
                l for l in reversed(discriminator.layers)
                if isinstance(l, tf.keras.layers.Flatten)
            )
            disc_features = tf.keras.Model(
                inputs=discriminator.input, outputs=feat_layer.output, name="disc_features"
            )
            scores = self._ev.compute_scores_fanogan(encoder, generator, disc_features, x)
            return self._full_eval(scores, y_test, "f-AnoGAN", "fanogan",
                                   heatmap_model=_ReconWrapper(encoder, generator), heatmap_x=x)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── Ablation evaluation ───────────────────────────────────────────────────

    _ABLATION_VARIANTS = {
        "full":     "improved_bigan",
        "no_skip":  "bigan_no_skip",
        "no_se":    "bigan_no_se",
        "fewer_se": "bigan_fewer_se",
    }

    def eval_ablation_bigan(self, variant: str, x_test, y_test):
        try:
            prefix = self._ABLATION_VARIANTS[variant]
            m = self._try_load(f"{prefix}_encoder.keras", f"{prefix}_generator.keras")
            if m is None:
                return None
            encoder, generator = m
            x = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
            scores = self._ev.compute_scores_bigan(encoder, generator, x)
            name = f"BiGAN-{variant}"
            out = Path(self._ablation_cfg.root_dir) / name
            return self._full_eval(scores, y_test, name, "bigan",
                                   heatmap_model=_ReconWrapper(encoder, generator),
                                   heatmap_x=x, out_dir=out)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── T-SNE ─────────────────────────────────────────────────────────────────

    def run_tsne(self, x_test: np.ndarray, y_test: np.ndarray):
        try:
            x_4d   = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
            x_flat = x_test.reshape(len(x_test), -1)
            latent_dict = {}
            ablation_latent_dict = {}

            def _tsne(name, fn, save_dir):
                try:
                    latent = fn()
                    if latent.ndim > 2:
                        latent = latent.reshape(len(latent), -1)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    self._viz.plot_tsne(latent, y_test, name, save_dir)
                    return latent
                except Exception as e:
                    logger.warning(f"T-SNE skipped for {name}: {e}")
                    return None

            for name, fname, x in [
                ("FC-AE",    "FC-AE.keras",    x_flat),
                ("CNN-AE",   "CNN-AE.keras",   x_4d),
                ("VAE",      "VAE.keras",      x_flat),
                ("Beta-VAE", "Beta-VAE.keras", x_flat),
                ("VQ-VAE",   "VQ-VAE.keras",   x_4d),
            ]:
                loaded = self._try_load(fname)
                if loaded is None:
                    continue
                _m, _x = loaded[0], x
                lat = _tsne(name, lambda _m=_m, _x=_x: _m.predict(_x, verbose=0),
                            self._out_dir(name))
                if lat is not None:
                    latent_dict[name] = lat

            loaded = self._try_load("CVAE.keras")
            if loaded is not None:
                cond = np.zeros((len(x_flat), N_CONDITIONS), dtype=np.float32)
                _m = loaded[0]
                lat = _tsne("CVAE", lambda _m=_m: _m.predict([x_flat, cond], verbose=0),
                            self._out_dir("CVAE"))
                if lat is not None:
                    latent_dict["CVAE"] = lat

            for name, enc_file in [
                ("BiGAN",          "bigan_encoder.keras"),
                ("Improved-BiGAN", "improved_bigan_encoder.keras"),
                ("GANomaly",       "ganomaly_encoder1.keras"),
                ("f-AnoGAN",       "fanogan_encoder.keras"),
            ]:
                loaded = self._try_load(enc_file)
                if loaded is None:
                    continue
                _enc = loaded[0]
                lat = _tsne(name, lambda _enc=_enc: _enc.predict(x_4d, verbose=0),
                            self._out_dir(name))
                if lat is not None:
                    latent_dict[name] = lat

            for variant, prefix in self._ABLATION_VARIANTS.items():
                loaded = self._try_load(f"{prefix}_encoder.keras")
                if loaded is None:
                    continue
                _enc = loaded[0]
                abl_name = f"BiGAN-{variant}"
                abl_out = Path(self._ablation_cfg.root_dir) / abl_name
                lat = _tsne(abl_name, lambda _enc=_enc: _enc.predict(x_4d, verbose=0), abl_out)
                if lat is not None:
                    ablation_latent_dict[abl_name] = lat

            if len(latent_dict) >= 2:
                comp_dir = Path(self._eval_cfg.comparison_dir)
                comp_dir.mkdir(parents=True, exist_ok=True)
                self._viz.plot_tsne_comparison(latent_dict, y_test, comp_dir)

            if len(ablation_latent_dict) >= 2:
                abl_comp = Path(self._ablation_cfg.comparison_dir)
                abl_comp.mkdir(parents=True, exist_ok=True)
                self._viz.plot_tsne_comparison(ablation_latent_dict, y_test, abl_comp)

            logger.info("T-SNE plots complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        try:
            x_test, y_test = self._load_test_data()
            results = {}

            for method, label in [
                (self.eval_fc_ae,        "FC-AE"),
                (self.eval_cnn_ae,       "CNN-AE"),
                (self.eval_vae,          "VAE"),
                (self.eval_beta_vae,     "Beta-VAE"),
                (self.eval_cvae,         "CVAE"),
                (self.eval_vqvae,        "VQ-VAE"),
                (self.eval_normal_bigan, "BiGAN"),
                (self.eval_bigan,        "Improved-BiGAN"),
                (self.eval_ganomaly,     "GANomaly"),
                (self.eval_fanogan,      "f-AnoGAN"),
            ]:
                logger.info(f"Evaluating: {label}")
                r = method(x_test, y_test)
                if r is not None:
                    results[label] = r

            if len(results) >= 2:
                comp_dir = Path(self._eval_cfg.comparison_dir)
                comp_dir.mkdir(parents=True, exist_ok=True)
                self._viz.plot_comparison(results, comp_dir)
                self._viz.plot_roc_comparison(results, comp_dir)
                logger.info(f"Comparison plots saved → {comp_dir}")

            logger.info("Ablation Study evaluation")
            ablation_results = {}
            for variant in self._ABLATION_VARIANTS:
                name = f"BiGAN-{variant}"
                logger.info(f"Ablation: {name}")
                r = self.eval_ablation_bigan(variant, x_test, y_test)
                if r is not None:
                    ablation_results[name] = r

            if len(ablation_results) >= 2:
                abl_comp = Path(self._ablation_cfg.comparison_dir)
                abl_comp.mkdir(parents=True, exist_ok=True)
                self._viz.plot_comparison(ablation_results, abl_comp)
                self._viz.plot_roc_comparison(ablation_results, abl_comp)
                logger.info(f"Ablation comparison plots saved → {abl_comp}")

            logger.info("Generating T-SNE plots...")
            self.run_tsne(x_test, y_test)

            return results
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e


if __name__ == "__main__":
    root = ConfigurationManager()
    all_datasets = list(root.config["data_ingestion"]["datasets"].keys())
    for ds in all_datasets:
        logger.info(f"{'='*60}")
        logger.info(f"Dataset: {ds}")
        logger.info(f"{'='*60}")
        EvaluationPipeline(dataset_name=ds).run()
