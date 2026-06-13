import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import IMG_SIZE

input_dim = IMG_SIZE[0] * IMG_SIZE[1]
# Custom VAE loss layer
class VAELossLayer(Layer):
    def __init__(self, beta=1.0, **kwargs):
        self.beta = beta
        super(VAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, x_decoded, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_decoded)) * input_dim
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + self.beta * kl_loss
        self.add_loss(total_loss)
        return x_decoded

# 3. Variational Autoencoder
# sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Parameters
latent_dim = 128

# encoder
inputs_vae = Input(shape=(input_dim,))
x = Dense(1024, activation='relu')(inputs_vae)
x = Dense(512, activation='relu')(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)
z = Lambda(sampling)([z_mean, z_log_var])

# decoder
x = Dense(512, activation='relu')(z)
x = Dense(1024, activation='relu')(x)
outputs_vae = Dense(input_dim, activation='sigmoid')(x)

# custom loss layer
vae_output = VAELossLayer(beta=0.01)([inputs_vae, outputs_vae, z_mean, z_log_var])

# VAE model
vae = Model(inputs_vae, vae_output)
vae.compile(optimizer='adam')