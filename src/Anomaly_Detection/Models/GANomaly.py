import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Flatten, Reshape, LeakyReLU, BatchNormalization,
    Conv2D, Conv2DTranspose, Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from Anomaly_Detection.constant import IMG_SHAPE, LATENT_DIM, LEARNING_RATE


def build_ganomaly_encoder(
    img_shape: tuple = IMG_SHAPE,
    latent_dim: int = LATENT_DIM,
    name: str = "ganomaly_encoder",
) -> Model:
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
    z = Dense(latent_dim, name="z")(x)
    return Model(inputs, z, name=name)


def build_ganomaly_generator(
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
    return Model(inputs, outputs, name="ganomaly_generator")


def build_ganomaly_discriminator(img_shape: tuple = IMG_SHAPE) -> Model:
    inputs = Input(shape=img_shape)
    x = Conv2D(32, (4, 4), strides=2, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(128, (4, 4), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    validity = Dense(1, activation='sigmoid')(x)
    return Model(inputs, validity, name="ganomaly_discriminator")


def build_ganomaly_model(
    encoder1: Model,
    generator: Model,
    encoder2: Model,
    discriminator: Model,
    img_shape: tuple = IMG_SHAPE,
    lr: float = LEARNING_RATE,
) -> Model:
    """Combined model: trains encoder1+generator+encoder2 with discriminator frozen."""
    discriminator.trainable = False
    img_input = Input(shape=img_shape)
    z1 = encoder1(img_input)
    x_hat = generator(z1)
    z2 = encoder2(x_hat)
    validity = discriminator(x_hat)
    model = Model(img_input, [validity, x_hat, z1, z2], name="ganomaly")
    model.compile(
        optimizer=Adam(learning_rate=lr, beta_1=0.5),
        loss=['binary_crossentropy', 'mse', 'mse', 'mse'],
        loss_weights=[1.0, 50.0, 1.0, 1.0],
    )
    return model
