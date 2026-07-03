from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import INPUT_DIM, LATENT_DIM, BETA_VALUE
from Anomaly_Detection.Models.VAE import VAELossLayer, sampling


def build_beta_vae(
    input_dim: int = INPUT_DIM,
    latent_dim: int = LATENT_DIM,
    beta: float = BETA_VALUE,
) -> Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(512, activation='relu')(x)
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, name="z_sampling")([z_mean, z_log_var])

    x = Dense(512, activation='relu')(z)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(input_dim, activation='sigmoid')(x)

    beta_vae_output = VAELossLayer(beta=beta, input_dim=input_dim)(
        [inputs, outputs, z_mean, z_log_var]
    )

    model = Model(inputs, beta_vae_output, name="beta_vae")
    model.compile(optimizer='adam')
    return model
