from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import IMG_SIZE

img_height, img_width = IMG_SIZE

# 6. VQ-VAE
# encoder
inputs_vq = Input(shape=(img_height, img_width, 1))

# Encoder
x = Conv2D(32, (3, 3), strides=2, padding='same')(inputs_vq)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

# Bottleneck
encoded_vq = Conv2D(64, (1, 1), padding='same')(x)
encoded_vq = BatchNormalization()(encoded_vq)
encoded_vq = LeakyReLU(0.2)(encoded_vq)

# Decoder
x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(encoded_vq)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

decoded_vq = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


vqvae = Model(inputs_vq, decoded_vq)
vqvae.compile(optimizer='adam', loss='mse')