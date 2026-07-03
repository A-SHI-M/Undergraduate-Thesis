import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose, Layer,
)
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import IMG_SHAPE


@tf.keras.utils.register_keras_serializable(package="AnomalyDetection")
class VectorQuantizer(Layer):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 commitment_cost: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name="embeddings",
            shape=(self.embedding_dim, self.num_embeddings),
            initializer="uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x: (batch, h, w, embedding_dim)
        flat = tf.reshape(x, [-1, self.embedding_dim])  # (N, D)

        # ||ze - ek||² = ||ze||² + ||ek||² - 2·ze·ek
        distances = (
            tf.reduce_sum(flat ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0, keepdims=True)
            - 2.0 * tf.matmul(flat, self.embeddings)
        )  # (N, num_embeddings)

        encoding_indices = tf.argmin(distances, axis=1)          # (N,)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)  # (N, K)
        quantized_flat = tf.matmul(encodings, tf.transpose(self.embeddings))  # (N, D)
        quantized = tf.reshape(quantized_flat, tf.shape(x))      # (batch, h, w, D)

        # Codebook loss: move embeddings toward encoder outputs
        codebook_loss = tf.reduce_mean((tf.stop_gradient(x) - quantized) ** 2)
        # Commitment loss: move encoder outputs toward embeddings
        commitment_loss = tf.reduce_mean((x - tf.stop_gradient(quantized)) ** 2)
        self.add_loss(codebook_loss + self.commitment_cost * commitment_loss)

        # Straight-through estimator: copy gradients from quantized to x
        return x + tf.stop_gradient(quantized - x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_embeddings": self.num_embeddings,
            "embedding_dim": self.embedding_dim,
            "commitment_cost": self.commitment_cost,
        })
        return config


def build_vqvae(
    img_shape: tuple = IMG_SHAPE,
    num_embeddings: int = 128,
    embedding_dim: int = 64,
    commitment_cost: float = 0.25,
) -> Model:
    inputs = Input(shape=img_shape)

    # Encoder
    x = Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(embedding_dim, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Vector Quantization bottleneck
    x = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost, name="vq")(x)

    # Decoder
    x = Conv2DTranspose(64, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(32, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2DTranspose(16, 3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    outputs = Conv2D(1, 3, activation='sigmoid', padding='same')(x)

    model = Model(inputs, outputs, name="vqvae")
    model.compile(optimizer='adam', loss='mse')
    return model
