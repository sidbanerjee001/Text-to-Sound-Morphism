import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# CAN WE MAKE A MEANINGFUL MAPPING BETWEEN TEXT AND MUSIC WITH UNSUPERVISED LEARNING?

# WAYS TO VALIDATE CLUSTERS: INTRA CLUSTER DISTANCE VS INTER CLUSTER, BIC MEASURE
# Geometry (differences) vs. topology (variances. may look different but encode the same concept)
# KL Divergence for a distribution that sums to 1 (i.e. FFT distribution, PSD)

X = np.load("processed_sound_files/features.npy")
y = np.load("processed_sound_files/labels.npy")

unique_labels = np.unique(y)
color_palette = sns.color_palette("coolwarm", len(unique_labels))
label_to_color = {label: color for label, color in zip(unique_labels, color_palette)}
colors = [label_to_color[label] for label in y]  # Map colors to labels

input_dim = X.shape[1]

# Autoencoder stuff
encoder_input = layers.Input(shape=(input_dim,))
encoder_output = layers.Dense(64, activation="relu")(encoder_input)
encoder_output = layers.Dense(32, activation="relu")(encoder_output)
encoder_output = layers.Dense(16, activation="relu")(encoder_output)
encoder = keras.Model(encoder_input, encoder_output, name="encoder")

decoder_input = layers.Input(shape=(16,))
decoder_output = layers.Dense(32, activation="relu")(decoder_input)
decoder_output = layers.Dense(64, activation="relu")(decoder_output)
decoder_output = layers.Dense(input_dim, activation="relu")(decoder_output)
decoder = keras.Model(decoder_input, decoder_output, name="decoder")

autoencoder_input = layers.Input(shape=(input_dim,))
encoded = encoder(autoencoder_input)
decoded = decoder(encoded)
autoencoder = keras.Model(autoencoder_input, decoded, name="autoencoder")

autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(X, X, epochs=50, batch_size=16)
X_encoded = encoder.predict(X)

# TRY t-SNE or t-SNE(2)

pca = PCA(n_components=2)
X_encoded_pca = pca.fit_transform(X_encoded)

# PLOT STUFF BELOW
fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    X_encoded_pca[:, 0], 
    X_encoded_pca[:, 1], 
    color=colors,
    alpha=0.7
)

ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("2D PCA Clustering of Encoded Music Features")

# Create legend
handles = [plt.Line2D([0], [0], marker="o", color="w", markersize=10, 
                      markerfacecolor=color) for color in color_palette]
ax.legend(handles, unique_labels, title="Instruments")

plt.tight_layout()
plt.show()