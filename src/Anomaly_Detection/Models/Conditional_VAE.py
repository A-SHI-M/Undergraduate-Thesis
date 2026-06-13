import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Concatenate, Dense, Lambda
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import IMG_SIZE

input_dim = IMG_SIZE[0] * IMG_SIZE[1]
n_conditions = 3

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# 5. Conditional VAE
# Custom CVAE loss layer
class CVAELossLayer(Layer):
    def __init__(self, beta=1.0, **kwargs):
        self.beta = beta
        super(CVAELossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        x, x_decoded, z_mean, z_log_var = inputs
        reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(x, x_decoded)) * input_dim
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        total_loss = reconstruction_loss + self.beta * kl_loss
        self.add_loss(total_loss)
        return x_decoded

# Parameters
latent_dim_cvae = 128

# encoder with condition
inputs_cvae = Input(shape=(input_dim,))
condition_input = Input(shape=(n_conditions,))
x = Concatenate()([inputs_cvae, condition_input])
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
z_mean_cvae = Dense(latent_dim_cvae)(x)
z_log_var_cvae = Dense(latent_dim_cvae)(x)
z_cvae = Lambda(sampling)([z_mean_cvae, z_log_var_cvae])

# decoder with condition
z_cond = Concatenate()([z_cvae, condition_input])
x = Dense(512, activation='relu')(z_cond)
x = Dense(1024, activation='relu')(x)
outputs_cvae = Dense(input_dim, activation='sigmoid')(x)

# custom loss layer
cvae_output = CVAELossLayer()([inputs_cvae, outputs_cvae, z_mean_cvae, z_log_var_cvae])

# CVAE model
cvae = Model([inputs_cvae, condition_input], cvae_output)
cvae.compile(optimizer='adam')