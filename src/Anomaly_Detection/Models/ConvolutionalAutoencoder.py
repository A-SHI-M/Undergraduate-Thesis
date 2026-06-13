from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Conv2DTranspose
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import IMG_SIZE

img_height, img_width = IMG_SIZE

# 2. Convolutional Autoencoder
input_cnn = Input(shape=(img_height, img_width, 1))


x = Conv2D(32, (3, 3), strides=2, padding='same')(input_cnn)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)


encoded_cnn = x


x = Conv2DTranspose(128, (3, 3), strides=2, padding='same')(encoded_cnn)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)

decoded_cnn = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder_cnn = Model(input_cnn, decoded_cnn)
autoencoder_cnn.compile(optimizer='adam', loss='mse')