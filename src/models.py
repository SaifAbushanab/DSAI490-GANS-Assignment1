import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

from .config import LATENT_DIM


def build_autoencoder():

    encoder = Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(LATENT_DIM)
    ])

    decoder = Sequential([
        layers.Input(shape=(LATENT_DIM,)),

        layers.Dense(8 * 8 * 64, activation="relu"),
        layers.Reshape((8, 8, 64)),

        layers.Conv2DTranspose(
            64, 3, strides=2,
            padding="same",
            activation="relu"
        ),

        layers.Conv2DTranspose(
            32, 3, strides=2,
            padding="same",
            activation="relu"
        ),

        layers.Conv2D(
            3, 3,
            padding="same",
            activation="sigmoid"
        )
    ])

    inputs = layers.Input(shape=(32, 32, 3))
    outputs = decoder(encoder(inputs))

    model = Model(inputs, outputs)

    return model, encoder