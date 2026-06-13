from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import IMG_SIZE


input_dim = IMG_SIZE[0] * IMG_SIZE[1]
# 1. Fully Connected Autoencoder
input_fc = Input(shape=(input_dim,))
encoded_fc = Dense(1024, activation='relu')(input_fc)
encoded_fc = Dense(512, activation='relu')(encoded_fc)
encoded_fc = Dense(256, activation='relu')(encoded_fc)


encoded_fc = Dense(128, activation='relu')(encoded_fc)


decoded_fc = Dense(256, activation='relu')(encoded_fc)
decoded_fc = Dense(512, activation='relu')(decoded_fc)
decoded_fc = Dense(1024, activation='relu')(decoded_fc)
decoded_fc = Dense(input_dim, activation='sigmoid')(decoded_fc)

autoencoder_fc = Model(input_fc, decoded_fc)
autoencoder_fc.compile(optimizer='adam', loss='mse')