import numpy as np
from pathlib import Path
from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.data_transformation import DataTransformation
from Anomaly_Detection.components.model_trainer import (
    ModelTrainer, BiGANTrainer, GANomalyTrainer, GANomalyTrainerConfig,
    fAnoGANTrainer, fAnoGANTrainerConfig,
)
from Anomaly_Detection.components.model_evaluation import ModelEvaluator, EvaluationResult
from Anomaly_Detection.utils.visualizations import Visualizer
from Anomaly_Detection.Models import (
    build_fc_autoencoder,
    build_cnn_autoencoder,
    build_vae,
    build_beta_vae,
    build_cvae,
    build_vqvae,
    build_encoder,
    build_generator,
    build_discriminator,
    build_bigan,
    build_reconstruction_model,
    build_ganomaly_encoder,
    build_ganomaly_generator,
    build_ganomaly_discriminator,
    build_ganomaly_model,
    build_fanogan_generator,
    build_fanogan_discriminator,
    build_fanogan_dcgan,
    build_fanogan_encoder,
)
from Anomaly_Detection.constant import INPUT_DIM, N_CONDITIONS, LATENT_DIM


ABLATION_VARIANTS = {
    "full":     dict(use_skip=True,  use_se=True,  fewer_se=False),
    "no_skip":  dict(use_skip=False, use_se=True,  fewer_se=False),
    "no_se":    dict(use_skip=True,  use_se=False, fewer_se=False),
    "fewer_se": dict(use_skip=True,  use_se=True,  fewer_se=True),
}


class TrainingPipeline:
    def __init__(self, dataset_name: str = None):
        self.cfg = ConfigurationManager(dataset_name=dataset_name)
        self._eval_cfg = self.cfg.get_model_evaluation_config()
        self._ablation_cfg = self.cfg.get_ablation_config()
        self._data_cfg = self.cfg.get_data_ingestion_config()
        self._bigan_cfg = self.cfg.get_bigan_trainer_config()
        self._viz = Visualizer()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_data(self):
        return DataTransformation(self._data_cfg).initiate()

    def _trainer(self, model_name: str) -> ModelTrainer:
        return ModelTrainer(self.cfg.get_model_trainer_config(model_name))

    def _evaluator(self) -> ModelEvaluator:
        return ModelEvaluator(self._eval_cfg)

    def _model_dir(self, name: str) -> Path:
        d = Path(self._eval_cfg.root_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _ablation_dir(self, name: str) -> Path:
        d = Path(self._ablation_cfg.root_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _evaluate_and_save(
        self,
        model,
        x_test: np.ndarray,
        y_test: np.ndarray,
        name: str,
        output_dir: Path,
        cvae_cond: np.ndarray = None,
    ) -> EvaluationResult:
        ev = self._evaluator()
        if cvae_cond is not None:
            scores = ev.compute_scores_cvae(model, x_test, cvae_cond)
        else:
            scores = ev.compute_scores_autoencoder(model, x_test)
        result = ev.evaluate(scores, y_test)
        ev.print_metrics(result, name)
        ev.save_metrics(result, name, output_dir)
        self._viz.plot_roc_curve(result, name, output_dir)
        self._viz.plot_pr_curve(result, name, output_dir)
        self._viz.plot_confusion_matrix(result, name, output_dir)
        self._viz.plot_score_distribution(result, name, output_dir)
        normal_idx = y_test == 0
        tumor_idx = y_test == 1
        self._viz.plot_heatmap(model, x_test[normal_idx], name, output_dir, label="normal",
                               cvae_conditions=cvae_cond[normal_idx] if cvae_cond is not None else None)
        self._viz.plot_heatmap(model, x_test[tumor_idx], name, output_dir, label="tumor",
                               cvae_conditions=cvae_cond[tumor_idx] if cvae_cond is not None else None)
        return result

    # ── Standard models ───────────────────────────────────────────────────────

    def run_fc_ae(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_tr = x_train.reshape(len(x_train), INPUT_DIM)
        x_te = x_test.reshape(len(x_test), INPUT_DIM)
        model = build_fc_autoencoder()
        self._trainer("fc_ae").train(model, x_tr, "FC-AE")
        return self._evaluate_and_save(model, x_te, y_test, "FC-AE", self._model_dir("FC-AE"))

    def run_cnn_ae(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_tr = x_train[..., np.newaxis]
        x_te = x_test[..., np.newaxis]
        model = build_cnn_autoencoder()
        self._trainer("cnn_ae").train(model, x_tr, "CNN-AE")
        return self._evaluate_and_save(model, x_te, y_test, "CNN-AE", self._model_dir("CNN-AE"))

    def run_vae(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_tr = x_train.reshape(len(x_train), INPUT_DIM)
        x_te = x_test.reshape(len(x_test), INPUT_DIM)
        p = self.cfg.params["vae"]
        model = build_vae(latent_dim=int(p["latent_dim"]), beta=float(p["beta"]))
        self._trainer("vae").train(model, x_tr, "VAE")
        return self._evaluate_and_save(model, x_te, y_test, "VAE", self._model_dir("VAE"))

    def run_beta_vae(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_tr = x_train.reshape(len(x_train), INPUT_DIM)
        x_te = x_test.reshape(len(x_test), INPUT_DIM)
        p = self.cfg.params["beta_vae"]
        model = build_beta_vae(latent_dim=int(p["latent_dim"]), beta=float(p["beta"]))
        self._trainer("beta_vae").train(model, x_tr, "Beta-VAE")
        return self._evaluate_and_save(model, x_te, y_test, "Beta-VAE", self._model_dir("Beta-VAE"))

    def run_cvae(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_tr = x_train.reshape(len(x_train), INPUT_DIM)
        x_te = x_test.reshape(len(x_test), INPUT_DIM)
        p = self.cfg.params["cvae"]
        n_cond = int(p["n_conditions"])
        cond_tr = np.zeros((len(x_tr), n_cond), dtype=np.float32)
        cond_te = np.zeros((len(x_te), n_cond), dtype=np.float32)
        model = build_cvae(latent_dim=int(p["latent_dim"]), n_conditions=n_cond)
        cfg = self.cfg.get_model_trainer_config("cvae")
        model.fit([x_tr, cond_tr], x_tr, epochs=cfg.epochs, batch_size=cfg.batch_size,
                  validation_split=cfg.validation_split, verbose=1)
        for save_dir in (cfg.models_dir, cfg.extra_models_dir):
            model.save(str(save_dir / "CVAE.keras"))
        return self._evaluate_and_save(model, x_te, y_test, "CVAE",
                                       self._model_dir("CVAE"), cvae_cond=cond_te)

    def run_vqvae(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_tr = x_train[..., np.newaxis]
        x_te = x_test[..., np.newaxis]
        p = self.cfg.params["vqvae"]
        model = build_vqvae(num_embeddings=int(p["num_embeddings"]),
                            embedding_dim=int(p["embedding_dim"]),
                            commitment_cost=float(p["commitment_cost"]))
        self._trainer("vqvae").train(model, x_tr, "VQ-VAE")
        return self._evaluate_and_save(model, x_te, y_test, "VQ-VAE", self._model_dir("VQ-VAE"))

    # ── BiGAN variants ────────────────────────────────────────────────────────

    def run_bigan(self, variant: str = "full", output_dir: Path = None) -> EvaluationResult:
        flags = ABLATION_VARIANTS[variant]
        x_train, x_test, y_test = self._load_data()
        x_te = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test

        encoder = build_encoder(**{k: flags[k] for k in ("use_se", "fewer_se")})
        generator = build_generator(**flags)
        discriminator = build_discriminator()
        bigan = build_bigan(generator, discriminator, encoder)
        reconstruction_model = build_reconstruction_model(encoder, generator)

        prefix = "improved_bigan" if variant == "full" else f"bigan_{variant}"
        BiGANTrainer(self._bigan_cfg).train(
            encoder, generator, discriminator, bigan, reconstruction_model, x_train,
            prefix=prefix,
        )

        name = "Improved-BiGAN" if variant == "full" else f"BiGAN-{variant}"
        out = output_dir or self._model_dir(name)
        ev = self._evaluator()
        scores = ev.compute_scores_bigan(encoder, generator, x_te)
        result = ev.evaluate(scores, y_test)
        ev.print_metrics(result, name)
        ev.save_metrics(result, name, out)
        self._viz.plot_roc_curve(result, name, out)
        self._viz.plot_pr_curve(result, name, out)
        self._viz.plot_confusion_matrix(result, name, out)
        self._viz.plot_score_distribution(result, name, out)
        self._viz.plot_heatmap(reconstruction_model, x_te[y_test == 0], name, out, label="normal")
        self._viz.plot_heatmap(reconstruction_model, x_te[y_test == 1], name, out, label="tumor")
        return result

    def run_normal_bigan(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_te = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test

        # Normal BiGAN = no SE, no skip
        encoder = build_encoder(use_se=False, fewer_se=False)
        generator = build_generator(use_skip=False, use_se=False, fewer_se=False)
        discriminator = build_discriminator()
        bigan = build_bigan(generator, discriminator, encoder)
        reconstruction_model = build_reconstruction_model(encoder, generator)

        p = self.cfg.params["bigan_normal"]
        from Anomaly_Detection.entity.config_entity import BiGANTrainerConfig
        from Anomaly_Detection.config.configuaration import ConfigurationManager
        trainer_cfg = self.cfg.get_bigan_trainer_config()
        from dataclasses import replace
        normal_cfg = BiGANTrainerConfig(
            root_dir=trainer_cfg.root_dir,
            models_dir=trainer_cfg.models_dir,
            extra_models_dir=trainer_cfg.extra_models_dir,
            pretrain_epochs=int(p["pretrain_epochs"]),
            bigan_epochs=int(p["bigan_epochs"]),
            disc_train_interval=int(p["disc_train_interval"]),
            learning_rate=float(p["learning_rate"]),
        )
        BiGANTrainer(normal_cfg).train(
            encoder, generator, discriminator, bigan, reconstruction_model, x_train,
            prefix="bigan",
        )

        name = "BiGAN"
        out = self._model_dir(name)
        ev = self._evaluator()
        scores = ev.compute_scores_bigan(encoder, generator, x_te)
        result = ev.evaluate(scores, y_test)
        ev.print_metrics(result, name)
        ev.save_metrics(result, name, out)
        self._viz.plot_roc_curve(result, name, out)
        self._viz.plot_pr_curve(result, name, out)
        self._viz.plot_confusion_matrix(result, name, out)
        self._viz.plot_score_distribution(result, name, out)
        self._viz.plot_heatmap(reconstruction_model, x_te[y_test == 0], name, out, label="normal")
        self._viz.plot_heatmap(reconstruction_model, x_te[y_test == 1], name, out, label="tumor")
        return result

    # ── GANomaly ──────────────────────────────────────────────────────────────

    def run_ganomaly(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_te = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test

        p = self.cfg.params["ganomaly"]
        trainer_cfg_base = self.cfg.get_model_trainer_config("fc_ae")
        trainer_cfg = GANomalyTrainerConfig(
            models_dir=trainer_cfg_base.models_dir,
            extra_models_dir=trainer_cfg_base.extra_models_dir,
            epochs=int(p["epochs"]),
            batch_size=int(p["batch_size"]),
            learning_rate=float(p["learning_rate"]),
        )

        encoder1 = build_ganomaly_encoder(name="ganomaly_encoder1")
        encoder2 = build_ganomaly_encoder(name="ganomaly_encoder2")
        generator = build_ganomaly_generator()
        discriminator = build_ganomaly_discriminator()
        ganomaly = build_ganomaly_model(encoder1, generator, encoder2, discriminator)

        GANomalyTrainer(trainer_cfg).train(
            encoder1, generator, encoder2, discriminator, ganomaly,
            x_train,
        )

        name = "GANomaly"
        out = self._model_dir(name)
        ev = self._evaluator()
        scores = ev.compute_scores_ganomaly(encoder1, generator, encoder2, x_te)
        result = ev.evaluate(scores, y_test)
        ev.print_metrics(result, name)
        ev.save_metrics(result, name, out)
        self._viz.plot_roc_curve(result, name, out)
        self._viz.plot_pr_curve(result, name, out)
        self._viz.plot_confusion_matrix(result, name, out)
        self._viz.plot_score_distribution(result, name, out)
        return result

    # ── AnoGAN ────────────────────────────────────────────────────────────────

    def run_anogan(self) -> EvaluationResult:
        x_train, x_test, y_test = self._load_data()
        x_te = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test

        p = self.cfg.params["anogan"]
        trainer_cfg_base = self.cfg.get_model_trainer_config("fc_ae")
        trainer_cfg = fAnoGANTrainerConfig(
            models_dir=trainer_cfg_base.models_dir,
            extra_models_dir=trainer_cfg_base.extra_models_dir,
            epochs=int(p["epochs"]),
            encoder_epochs=int(p["encoder_epochs"]),
            batch_size=int(p["batch_size"]),
            learning_rate=float(p["learning_rate"]),
        )

        generator = build_fanogan_generator()
        discriminator = build_fanogan_discriminator()
        dcgan = build_fanogan_dcgan(generator, discriminator)
        encoder = build_fanogan_encoder()

        fAnoGANTrainer(trainer_cfg).train(
            generator, discriminator, dcgan, encoder, x_train,
        )

        import tensorflow as tf
        feat_layer = next(
            l for l in reversed(discriminator.layers)
            if isinstance(l, tf.keras.layers.Flatten)
        )
        disc_features = tf.keras.Model(
            inputs=discriminator.input,
            outputs=feat_layer.output,
            name="disc_features",
        )

        name = "f-AnoGAN"
        out = self._model_dir(name)
        ev = self._evaluator()
        scores = ev.compute_scores_fanogan(encoder, generator, disc_features, x_te)
        result = ev.evaluate(scores, y_test)
        ev.print_metrics(result, name)
        ev.save_metrics(result, name, out)
        self._viz.plot_roc_curve(result, name, out)
        self._viz.plot_pr_curve(result, name, out)
        self._viz.plot_confusion_matrix(result, name, out)
        self._viz.plot_score_distribution(result, name, out)
        return result

    # ── Ablation study ────────────────────────────────────────────────────────

    def run_ablation(self) -> dict:
        ablation_results = {}
        for variant in ABLATION_VARIANTS:
            print(f"\n{'='*50}\nAblation variant: {variant}\n{'='*50}")
            out = self._ablation_dir(f"bigan_{variant}")
            ablation_results[f"BiGAN-{variant}"] = self.run_bigan(variant=variant, output_dir=out)

        comp_dir = Path(self._ablation_cfg.comparison_dir)
        comp_dir.mkdir(parents=True, exist_ok=True)
        self._viz.plot_comparison(ablation_results, comp_dir)
        self._viz.plot_roc_comparison(ablation_results, comp_dir)
        return ablation_results

    # ── T-SNE latent space ────────────────────────────────────────────────────

    def run_tsne(self, model_store: dict, x_test: np.ndarray, y_test: np.ndarray):
        """
        model_store: dict of {model_name: model_or_encoder}
        Extracts latent vectors and produces per-model and comparison T-SNE plots.
        """
        latent_dict = {}
        x_4d = x_test[..., np.newaxis] if x_test.ndim == 3 else x_test
        x_flat = x_test.reshape(len(x_test), -1)

        for name, obj in model_store.items():
            try:
                if name in ("BiGAN", "Improved-BiGAN"):
                    encoder, _ = obj
                    latent = encoder.predict(x_4d, verbose=0)
                elif name == "GANomaly":
                    encoder1, _, _ = obj
                    latent = encoder1.predict(x_4d, verbose=0)
                elif name in ("FC-AE", "VAE", "Beta-VAE"):
                    latent = obj.predict(x_flat, verbose=0)
                else:
                    latent = obj.predict(x_4d, verbose=0)
                    latent = latent.reshape(len(latent), -1)

                out = self._model_dir(name)
                self._viz.plot_tsne(latent, y_test, name, out)
                latent_dict[name] = latent
            except Exception as e:
                print(f"T-SNE skipped for {name}: {e}")

        if latent_dict:
            comp_dir = Path(self._eval_cfg.comparison_dir)
            comp_dir.mkdir(parents=True, exist_ok=True)
            self._viz.plot_tsne_comparison(latent_dict, y_test, comp_dir)
