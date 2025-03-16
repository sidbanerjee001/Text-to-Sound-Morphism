import tensorflow as tf
from tensorflow.keras import Layer, layers, Model, backend as K

import numpy as np

mfcc_features = np.load("processed_sound_files/features.npy")
labels = np.load("processed_sound_files/labels.npy")

# Hyperparameters
input_dim = mfcc_features.shape[1]  # Dimension of MFCC features
latent_dim = 12 # Dimension of the latent space

# Encoder
encoder_inputs = layers.Input(shape=(input_dim,))
encoder_outputs = layers.Dense(128, activation="relu")(encoder_inputs)
encoder_outputs = layers.Dense(64, activation="relu")(encoder_inputs)
encoder_outputs = layers.Dense(32, activation="relu")(encoder_outputs)
encoder_outputs = layers.Dense(16, activation="relu")(encoder_outputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(encoder_outputs)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(encoder_outputs)

# Custom Sampling Layer
class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = Sampling(name="z")([z_mean, z_log_var])

# Define encoder model
encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder section
latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = layers.Dense(32, activation="relu")(latent_inputs)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
decoder_outputs = layers.Dense(input_dim, activation="linear")(x)

# Define decoder model
decoder = Model(latent_inputs, decoder_outputs, name="decoder")

# VAE model
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = Model(encoder_inputs, vae_outputs, name="vae")

def vae_loss(inputs, outputs):
    # Beta value for implementing a b-VAE
    beta = 1.2

    # reconstruction_loss = K.mean(K.square(inputs - outputs), axis=1)
    reconstruction_loss = K.mean(K.abs(inputs - outputs), axis=1)  # MAE
    
    # KL divergence loss
    z_mean, z_log_var = encoder(inputs)[0], encoder(inputs)[1]
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)

    total_loss = K.mean(reconstruction_loss + beta * kl_loss)
    return total_loss

vae.compile(optimizer="adam", loss=vae_loss)

# Train the VAE
vae.fit(mfcc_features, mfcc_features, epochs=100, batch_size=32, shuffle=True)
latent_vectors = encoder.predict(mfcc_features)[2]

np.save('model_outputs/vae_LV_output.npy', latent_vectors)
np.save('model_outputs/vae_LV_labels.npy', labels)

print("VAE output saved.")