import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import INPUT_DIM, LATENT_DIM, N_CONDITIONS


class CVAELossLayer(Layer):
    def __init__(self, beta: float = 1.0, input_dim: int = INPUT_DIM, **kwargs):
        self.beta = beta
        self.input_dim = input_dim
        super().__init__(**kwargs)

    def call(self, inputs):
        x, x_decoded, z_mean, z_log_var = inputs
        reconstruction_loss = (
            tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_decoded)) * self.input_dim
        )
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(reconstruction_loss + self.beta * kl_loss)
        return x_decoded

    def get_config(self):
        config = super().get_config()
        config.update({"beta": self.beta, "input_dim": self.input_dim})
        return config


def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_cvae(
    input_dim: int = INPUT_DIM,
    latent_dim: int = LATENT_DIM,
    n_conditions: int = N_CONDITIONS,
    beta: float = 1.0,
) -> Model:
    inputs = Input(shape=(input_dim,))
    condition = Input(shape=(n_conditions,))

    x = Concatenate()([inputs, condition])
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, name="z_sampling")([z_mean, z_log_var])

    z_cond = Concatenate()([z, condition])
    x = Dense(512, activation='relu')(z_cond)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(input_dim, activation='sigmoid')(x)

    cvae_output = CVAELossLayer(beta=beta, input_dim=input_dim)(
        [inputs, outputs, z_mean, z_log_var]
    )

    model = Model([inputs, condition], cvae_output, name="cvae")
    model.compile(optimizer='adam')
    return model
