import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# CAN WE MAKE A MEANINGFUL MAPPING BETWEEN TEXT AND MUSIC WITH UNSUPERVISED LEARNING?

# WAYS TO VALIDATE CLUSTERS: INTRA CLUSTER DISTANCE VS INTER CLUSTER, BIC MEASURE
# Geometry (differences) vs. topology (variances. may look different but encode the same concept)
# KL Divergence for a distribution that sums to 1 (i.e. FFT distribution, PSD)

X = np.load("processed_sound_files/sound_data.npy")
y = np.load("processed_sound_files/sound_labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.80, random_state=42)

input_dim = X.shape[1]

# Autoencoder stuff
encoder_input = layers.Input(shape=(input_dim,))
encoder_output = layers.Dense(512, activation="relu")(encoder_input)
encoder_output = layers.Dense(64, activation="relu")(encoder_output)
encoder_output = layers.Dense(14, activation="relu")(encoder_output)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")

decoder_input = layers.Input(shape=(14,))
decoder_output = layers.Dense(64, activation="relu")(decoder_input)
decoder_output = layers.Dense(512, activation="relu")(decoder_output)
decoder_output = layers.Dense(input_dim, activation="relu")(decoder_output)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")

autoencoder_input = layers.Input(shape=(input_dim,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(X_train, X_train, epochs=50, batch_size=16)
X_encoded = encoder.predict(X_test)

np.save('model_outputs/autoencoder_output.npy', X_encoded)
np.save('model_outputs/autoencoder_labels.npy', y_test)

print("Model output and labels saved.")