import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization,
    Conv2D, Conv2DTranspose, Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Anomaly_Detection.constant import IMG_SHAPE, LATENT_DIM, LEARNING_RATE


def build_fanogan_generator(
    latent_dim: int = LATENT_DIM,
    img_shape: tuple = IMG_SHAPE,
) -> Model:
    inputs = Input(shape=(latent_dim,))
    x = Dense(8 * 8 * 256)(inputs)
    x = LeakyReLU(0.2)(x)
    x = Reshape((8, 8, 256))(x)
    x = Conv2DTranspose(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(32, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(16, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    outputs = Conv2D(img_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)
    return Model(inputs, outputs, name="fanogan_generator")


def build_fanogan_discriminator(img_shape: tuple = IMG_SHAPE) -> Model:
    inputs = Input(shape=img_shape)
    x = Conv2D(32, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    features = Flatten()(x)
    x = Dropout(0.5)(features)
    validity = Dense(1, activation='sigmoid')(x)
    return Model(inputs, [validity, features], name="fanogan_discriminator")


def build_fanogan_dcgan(
    generator: Model,
    discriminator: Model,
    latent_dim: int = LATENT_DIM,
    lr: float = LEARNING_RATE,
) -> Model:
    """Adversarial model for training the generator; discriminator frozen."""
    discriminator.trainable = False
    z = Input(shape=(latent_dim,))
    img = generator(z)
    validity, _ = discriminator(img)
    model = Model(z, validity, name="fanogan_dcgan")
    model.compile(optimizer=Adam(learning_rate=lr, beta_1=0.5), loss='binary_crossentropy')
    return model


def build_fanogan_encoder(
    img_shape: tuple = IMG_SHAPE,
    latent_dim: int = LATENT_DIM,
) -> Model:
    """f-AnoGAN encoder: maps an image directly to latent z (single forward pass)."""
    inputs = Input(shape=img_shape)
    x = Conv2D(32, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(256, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    z = Dense(latent_dim)(x)
    return Model(inputs, z, name="fanogan_encoder")
