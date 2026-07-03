import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Anomaly_Detection.entity.config_entity import ModelTrainerConfig, BiGANTrainerConfig
from Anomaly_Detection.constant import IMG_SHAPE, LATENT_DIM
from Anomaly_Detection.Models.ImprovedBIGAN import ssim_loss


class GANomalyTrainerConfig:
    def __init__(self, models_dir, extra_models_dir, epochs, batch_size, learning_rate):
        self.models_dir = models_dir
        self.extra_models_dir = extra_models_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class fAnoGANTrainerConfig:
    def __init__(self, models_dir, extra_models_dir, epochs, encoder_epochs, batch_size, learning_rate):
        self.models_dir = models_dir
        self.extra_models_dir = extra_models_dir
        self.epochs = epochs
        self.encoder_epochs = encoder_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate


class ModelTrainer:
    """Trains any Keras autoencoder / VAE on normal images."""

    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(
        self,
        model: Model,
        x_train: np.ndarray,
        model_name: str,
        y_train: np.ndarray = None,
    ) -> tf.keras.callbacks.History:
        target = y_train if y_train is not None else x_train
        history = model.fit(
            x_train,
            target,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=self.config.validation_split,
            verbose=1,
        )
        for save_dir in (self.config.models_dir, self.config.extra_models_dir):
            path = save_dir / f"{model_name}.keras"
            model.save(str(path))
            print(f"Saved {model_name} → {path}")
        return history


class BiGANTrainer:
    """Two-phase trainer for the Improved BiGAN."""

    def __init__(self, config: BiGANTrainerConfig):
        self.config = config

    def _reconstruction_loss(self, x_true: tf.Tensor, x_pred: tf.Tensor) -> tf.Tensor:
        mse = tf.reduce_mean(tf.square(x_true - x_pred))
        mae = tf.reduce_mean(tf.abs(x_true - x_pred))
        ssim = ssim_loss(x_true, x_pred)
        return 0.6 * mse + 0.3 * mae + 0.1 * ssim

    def train(
        self,
        encoder: Model,
        generator: Model,
        discriminator: Model,
        bigan: Model,
        reconstruction_model: Model,
        x_train: np.ndarray,
        latent_dim: int = LATENT_DIM,
        prefix: str = "bigan",
    ) -> dict:
        n = len(x_train)
        lr = self.config.learning_rate
        recon_opt = Adam(learning_rate=lr)
        disc_opt = Adam(learning_rate=lr, beta_1=0.5)  # noqa: F841 — reserved for manual disc training

        losses: dict = {"pretrain_recon": [], "bigan": [], "recon": []}
        batch_size = 16   # kept fixed; expose via config if needed

        # ── Phase 1: pre-train reconstruction ────────────────────────────────
        print(f"\nPhase 1: pre-training reconstruction ({self.config.pretrain_epochs} epochs)")
        for epoch in range(self.config.pretrain_epochs):
            idx = np.random.randint(0, n, batch_size)
            real_tensor = tf.convert_to_tensor(
                x_train[idx][..., np.newaxis], dtype=tf.float32
            )
            with tf.GradientTape() as tape:
                reconstructed = reconstruction_model(real_tensor, training=True)
                loss = self._reconstruction_loss(real_tensor, reconstructed)
            grads = tape.gradient(loss, reconstruction_model.trainable_variables)
            recon_opt.apply_gradients(zip(grads, reconstruction_model.trainable_variables))
            losses["pretrain_recon"].append(float(loss))
            if (epoch + 1) % 10 == 0:
                print(f"  [P1 {epoch+1}/{self.config.pretrain_epochs}] loss={loss:.4f}")

        # ── Phase 2: joint BiGAN + reconstruction ────────────────────────────
        print(f"\nPhase 2: joint training ({self.config.bigan_epochs} epochs)")
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        for epoch in range(self.config.bigan_epochs):
            idx = np.random.randint(0, n, batch_size)
            real_imgs = x_train[idx][..., np.newaxis].astype(np.float32)

            if (epoch + 1) % self.config.disc_train_interval == 0:
                z_noise = np.random.normal(0, 1, (batch_size, latent_dim))
                fake_imgs = generator.predict(z_noise, verbose=0)
                real_z = encoder.predict(real_imgs, verbose=0)
                discriminator.trainable = True
                d_real = discriminator.train_on_batch([real_imgs, real_z], real_labels)
                d_fake = discriminator.train_on_batch([fake_imgs, z_noise], fake_labels)
                discriminator.trainable = False
                losses["bigan"].append(float(0.5 * (d_real + d_fake)))

            z_noise = np.random.normal(0, 1, (batch_size, latent_dim))
            bigan.train_on_batch([z_noise, real_imgs], [real_labels, fake_labels])

            for _ in range(3):
                real_tensor = tf.convert_to_tensor(real_imgs, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    reconstructed = reconstruction_model(real_tensor, training=True)
                    r_loss = self._reconstruction_loss(real_tensor, reconstructed)
                grads = tape.gradient(r_loss, reconstruction_model.trainable_variables)
                recon_opt.apply_gradients(zip(grads, reconstruction_model.trainable_variables))
            losses["recon"].append(float(r_loss))

            if (epoch + 1) % 50 == 0:
                print(f"  [P2 {epoch+1}/{self.config.bigan_epochs}] recon={r_loss:.4f}")

        self._save(encoder, generator, discriminator, prefix)
        return losses

    def _save(self, encoder: Model, generator: Model, discriminator: Model, prefix: str = "bigan"):
        for model, name in [
            (encoder, f"{prefix}_encoder"),
            (generator, f"{prefix}_generator"),
            (discriminator, f"{prefix}_discriminator"),
        ]:
            for save_dir in (self.config.models_dir, self.config.extra_models_dir):
                path = save_dir / f"{name}.keras"
                model.save(str(path))
                print(f"Saved {name} → {path}")


class GANomalyTrainer:
    def __init__(self, config: GANomalyTrainerConfig):
        self.config = config

    def train(
        self,
        encoder1: Model,
        generator: Model,
        encoder2: Model,
        discriminator: Model,
        ganomaly_model: Model,
        x_train: np.ndarray,
    ) -> dict:
        n = len(x_train)
        batch_size = self.config.batch_size
        disc_opt = Adam(learning_rate=self.config.learning_rate, beta_1=0.5)
        losses = {"disc": [], "ganomaly": []}

        print(f"\nTraining GANomaly ({self.config.epochs} epochs)")
        for epoch in range(self.config.epochs):
            idx = np.random.randint(0, n, batch_size)
            real_imgs = x_train[idx][..., np.newaxis].astype(np.float32)

            z1 = encoder1.predict(real_imgs, verbose=0)
            fake_imgs = generator.predict(z1, verbose=0)
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            discriminator.trainable = True
            d_real = discriminator.train_on_batch(real_imgs, real_labels)
            d_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
            discriminator.trainable = False
            losses["disc"].append(float(0.5 * (d_real + d_fake)))

            g_loss = ganomaly_model.train_on_batch(
                real_imgs,
                [real_labels, real_imgs, z1, z1],
            )
            losses["ganomaly"].append(float(g_loss[0]) if isinstance(g_loss, list) else float(g_loss))

            if (epoch + 1) % 50 == 0:
                print(f"  [{epoch+1}/{self.config.epochs}] d={losses['disc'][-1]:.4f}")

        for model, name in [(encoder1, "ganomaly_encoder1"), (generator, "ganomaly_generator"),
                            (encoder2, "ganomaly_encoder2"), (discriminator, "ganomaly_discriminator")]:
            for save_dir in (self.config.models_dir, self.config.extra_models_dir):
                path = save_dir / f"{name}.keras"
                model.save(str(path))
                print(f"Saved {name} → {path}")
        return losses


class fAnoGANTrainer:
    def __init__(self, config: fAnoGANTrainerConfig):
        self.config = config

    def train(
        self,
        generator: Model,
        discriminator: Model,
        dcgan: Model,
        encoder: Model,
        x_train: np.ndarray,
        latent_dim: int = LATENT_DIM,
        kappa: float = 0.1,
    ) -> dict:
        n = len(x_train)
        batch_size = self.config.batch_size
        losses = {"disc": [], "gen": [], "encoder": []}

        # ── Phase 1: Train DCGAN on normal images ────────────────────────────
        print(f"\nf-AnoGAN Phase 1: training DCGAN ({self.config.epochs} epochs)")
        for epoch in range(self.config.epochs):
            idx = np.random.randint(0, n, batch_size)
            real_imgs = x_train[idx][..., np.newaxis].astype(np.float32)
            z = np.random.normal(0, 1, (batch_size, latent_dim))
            fake_imgs = generator.predict(z, verbose=0)

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            discriminator.trainable = True
            d_real = discriminator.train_on_batch(real_imgs, [real_labels, real_labels])
            d_fake = discriminator.train_on_batch(fake_imgs, [fake_labels, fake_labels])
            discriminator.trainable = False
            losses["disc"].append(float(0.5 * (np.mean(d_real) + np.mean(d_fake))))

            g_loss = dcgan.train_on_batch(z, real_labels)
            losses["gen"].append(float(g_loss))

            if (epoch + 1) % 50 == 0:
                print(f"  [P1 {epoch+1}/{self.config.epochs}] d={losses['disc'][-1]:.4f} g={g_loss:.4f}")

        # ── Phase 2: Train encoder with GAN frozen ───────────────────────────
        print(f"\nf-AnoGAN Phase 2: training encoder ({self.config.encoder_epochs} epochs)")
        generator.trainable = False
        discriminator.trainable = False
        enc_opt = Adam(learning_rate=self.config.learning_rate, beta_1=0.5)

        # Build a feature extractor from the discriminator (layer before Dropout)
        feat_layer = next(l for l in reversed(discriminator.layers) if isinstance(l, tf.keras.layers.Flatten))
        disc_features = tf.keras.Model(
            inputs=discriminator.input,
            outputs=feat_layer.output,
            name="disc_feat",
        )

        for epoch in range(self.config.encoder_epochs):
            idx = np.random.randint(0, n, batch_size)
            real_imgs = tf.constant(x_train[idx][..., np.newaxis], dtype=tf.float32)

            with tf.GradientTape() as tape:
                z_hat = encoder(real_imgs, training=True)
                x_hat = generator(z_hat, training=False)
                recon_loss = tf.reduce_mean(tf.square(real_imgs - x_hat))
                feat_real = disc_features(real_imgs, training=False)
                feat_fake = disc_features(x_hat, training=False)
                feat_loss = tf.reduce_mean(tf.square(feat_real - feat_fake))
                enc_loss = (1 - kappa) * recon_loss + kappa * feat_loss

            grads = tape.gradient(enc_loss, encoder.trainable_variables)
            enc_opt.apply_gradients(zip(grads, encoder.trainable_variables))
            losses["encoder"].append(float(enc_loss))

            if (epoch + 1) % 50 == 0:
                print(f"  [P2 {epoch+1}/{self.config.encoder_epochs}] enc={float(enc_loss):.4f}")

        generator.trainable = True
        discriminator.trainable = True

        for model, name in [
            (generator, "fanogan_generator"),
            (discriminator, "fanogan_discriminator"),
            (encoder, "fanogan_encoder"),
        ]:
            for save_dir in (self.config.models_dir, self.config.extra_models_dir):
                path = save_dir / f"{name}.keras"
                model.save(str(path))
                print(f"Saved {name} → {path}")
        return losses
