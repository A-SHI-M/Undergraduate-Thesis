import sys
import numpy as np
from Anomaly_Detection.config.configuaration import ConfigurationManager
from Anomaly_Detection.components.data_transformation import DataTransformation
from Anomaly_Detection.components.model_trainer import (
    ModelTrainer, BiGANTrainer, GANomalyTrainer, GANomalyTrainerConfig,
    fAnoGANTrainer, fAnoGANTrainerConfig,
)
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
from Anomaly_Detection.logger import logger
from Anomaly_Detection.exception import AnomalyDetectionException


ABLATION_VARIANTS = {
    "full":     dict(use_skip=True,  use_se=True,  fewer_se=False),
    "no_skip":  dict(use_skip=False, use_se=True,  fewer_se=False),
    "no_se":    dict(use_skip=True,  use_se=False, fewer_se=False),
    "fewer_se": dict(use_skip=True,  use_se=True,  fewer_se=True),
}


class TrainingPipeline:
    def __init__(self, dataset_name: str = None):
        self.cfg = ConfigurationManager(dataset_name=dataset_name)
        self._data_cfg = self.cfg.get_data_transformation_config()
        self._bigan_cfg = self.cfg.get_bigan_trainer_config()
        _, self._models_dir, _ = self.cfg.get_model_dirs()

    def _load_data(self):
        if not hasattr(self, "_cached_data"):
            self._cached_data = DataTransformation(self._data_cfg).initiate()
        return self._cached_data

    def _trainer(self, model_name: str) -> ModelTrainer:
        return ModelTrainer(self.cfg.get_model_trainer_config(model_name))

    # ── Standard autoencoders ─────────────────────────────────────────────────

    def run_fc_ae(self):
        try:
            logger.info(">>> Training: FC Autoencoder")
            x_train, _, _ = self._load_data()
            x_tr = x_train.reshape(x_train.shape[0], -1)
            model = build_fc_autoencoder()
            self._trainer("fc_ae").train(model, x_tr, "FC-AE")
            logger.info("FC Autoencoder training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def run_cnn_ae(self):
        try:
            logger.info(">>> Training: CNN Autoencoder")
            x_train, _, _ = self._load_data()
            x_tr = x_train[..., np.newaxis]
            model = build_cnn_autoencoder()
            self._trainer("cnn_ae").train(model, x_tr, "CNN-AE")
            logger.info("CNN Autoencoder training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def run_vae(self):
        try:
            logger.info(">>> Training: VAE")
            x_train, _, _ = self._load_data()
            x_tr = x_train.reshape(x_train.shape[0], -1)
            p = self.cfg.params["vae"]
            model = build_vae(latent_dim=int(p["latent_dim"]), beta=float(p["beta"]))
            self._trainer("vae").train(model, x_tr, "VAE")
            logger.info("VAE training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def run_beta_vae(self):
        try:
            logger.info(">>> Training: Beta-VAE")
            x_train, _, _ = self._load_data()
            x_tr = x_train.reshape(x_train.shape[0], -1)
            p = self.cfg.params["beta_vae"]
            model = build_beta_vae(latent_dim=int(p["latent_dim"]), beta=float(p["beta"]))
            self._trainer("beta_vae").train(model, x_tr, "Beta-VAE")
            logger.info("Beta-VAE training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def run_cvae(self):
        try:
            logger.info(">>> Training: CVAE")
            x_train, _, _ = self._load_data()
            x_tr = x_train.reshape(x_train.shape[0], -1)
            p = self.cfg.params["cvae"]
            n_cond = int(p["n_conditions"])
            cond_tr = np.zeros((len(x_tr), n_cond), dtype=np.float32)
            model = build_cvae(latent_dim=int(p["latent_dim"]), n_conditions=n_cond)
            cfg = self.cfg.get_model_trainer_config("cvae")
            model.fit([x_tr, cond_tr], x_tr, epochs=cfg.epochs, batch_size=cfg.batch_size,
                      validation_split=cfg.validation_split, verbose=1)
            for save_dir in (cfg.models_dir, cfg.extra_models_dir):
                model.save(str(save_dir / "CVAE.keras"))
            logger.info("CVAE training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def run_vqvae(self):
        try:
            logger.info(">>> Training: VQ-VAE")
            x_train, _, _ = self._load_data()
            x_tr = x_train[..., np.newaxis]
            p = self.cfg.params["vqvae"]
            model = build_vqvae(num_embeddings=int(p["num_embeddings"]),
                                embedding_dim=int(p["embedding_dim"]),
                                commitment_cost=float(p["commitment_cost"]))
            self._trainer("vqvae").train(model, x_tr, "VQ-VAE")
            logger.info("VQ-VAE training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── BiGAN variants ────────────────────────────────────────────────────────

    def run_normal_bigan(self):
        try:
            logger.info(">>> Training: BiGAN (Normal)")
            x_train, _, _ = self._load_data()
            encoder = build_encoder(use_se=False, fewer_se=False)
            generator = build_generator(use_skip=False, use_se=False, fewer_se=False)
            discriminator = build_discriminator()
            bigan = build_bigan(generator, discriminator, encoder)
            reconstruction_model = build_reconstruction_model(encoder, generator)

            p = self.cfg.params["bigan_normal"]
            from Anomaly_Detection.entity.config_entity import BiGANTrainerConfig
            trainer_cfg = self.cfg.get_bigan_trainer_config()
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
            logger.info("BiGAN (Normal) training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    def run_bigan(self, variant: str = "full"):
        try:
            name = "Improved-BiGAN" if variant == "full" else f"BiGAN-{variant}"
            logger.info(f">>> Training: {name}")
            flags = ABLATION_VARIANTS[variant]
            x_train, _, _ = self._load_data()
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
            logger.info(f"{name} training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── GANomaly ──────────────────────────────────────────────────────────────

    def run_ganomaly(self):
        try:
            logger.info(">>> Training: GANomaly")
            x_train, _, _ = self._load_data()
            p = self.cfg.params["ganomaly"]
            _, models_dir, extra_models_dir = self.cfg.get_model_dirs()
            trainer_cfg = GANomalyTrainerConfig(
                models_dir=models_dir,
                extra_models_dir=extra_models_dir,
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
                encoder1, generator, encoder2, discriminator, ganomaly, x_train,
            )
            logger.info("GANomaly training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── f-AnoGAN ──────────────────────────────────────────────────────────────

    def run_fanogan(self):
        try:
            logger.info(">>> Training: f-AnoGAN")
            x_train, _, _ = self._load_data()
            p = self.cfg.params["fanogan"]
            _, models_dir, extra_models_dir = self.cfg.get_model_dirs()
            trainer_cfg = fAnoGANTrainerConfig(
                models_dir=models_dir,
                extra_models_dir=extra_models_dir,
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
            logger.info("f-AnoGAN training complete")
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e

    # ── Ablation study ────────────────────────────────────────────────────────

    def run_ablation(self):
        try:
            for variant in ABLATION_VARIANTS:
                logger.info(f"Ablation variant: {variant}")
                self.run_bigan(variant=variant)
        except Exception as e:
            raise AnomalyDetectionException(e, sys) from e


if __name__ == "__main__":
    root = ConfigurationManager()
    all_datasets = list(root.config["data_ingestion"]["datasets"].keys())
    for ds in all_datasets:
        logger.info(f"{'='*60}")
        logger.info(f"Dataset: {ds}")
        logger.info(f"{'='*60}")
        pipeline = TrainingPipeline(dataset_name=ds)
        pipeline.run_fc_ae()
        pipeline.run_cnn_ae()
        pipeline.run_vae()
        pipeline.run_beta_vae()
        pipeline.run_cvae()
        pipeline.run_vqvae()
        pipeline.run_ganomaly()
        pipeline.run_fanogan()
        pipeline.run_normal_bigan()
        pipeline.run_bigan()
        pipeline.run_ablation()
