from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from Anomaly_Detection.constant import INPUT_DIM


def build_fc_autoencoder(input_dim: int = INPUT_DIM) -> Model:
    inputs = Input(shape=(input_dim,))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)

    x = Dense(256, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(input_dim, activation='sigmoid')(x)

    model = Model(inputs, outputs, name="fc_autoencoder")
    model.compile(optimizer='adam', loss='mse')
    return model
