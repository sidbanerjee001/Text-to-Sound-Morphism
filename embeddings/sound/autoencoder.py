import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data
X = np.load("processed_sound_files/sound_data.npy")
y = np.load("processed_sound_files/sound_labels.npy")

X = X.squeeze(axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Reshape for Conv1D: (samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

input_dim = X.shape[1]

# Encoder
inputs = keras.Input(shape=(input_dim, 1))
x = layers.Conv1D(256, 5, activation="relu", padding="same")(inputs)
x = layers.MaxPooling1D(2)(x)  # Downsampling
x = layers.BatchNormalization()(x)

x = layers.Conv1D(128, 5, activation="relu", padding="same")(x)
x = layers.MaxPooling1D(2)(x)  # Further downsampling
x = layers.BatchNormalization()(x)

x = layers.Flatten()(x)

# Add dropout to prevent overfitting
x = layers.Dropout(0.3)(x)

# Dense layers with regularization
x = layers.Dense(64, activation="relu", 
                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

# Latent space with linear activation and regularization
latent = layers.Dense(14, activation="linear", 
                        kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

# Decoder
x = layers.Dense(64, activation="relu", 
                    kernel_regularizer=tf.keras.regularizers.l2(0.001))(latent)

x = layers.Dense(input_dim * 128, activation="relu")(x)
x = layers.Reshape((input_dim, 128))(x)

x = layers.Conv1DTranspose(128, 5, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)

x = layers.Conv1DTranspose(256, 5, activation="relu", padding="same")(x)
x = layers.BatchNormalization()(x)

outputs = layers.Conv1DTranspose(1, 5, activation="linear", padding="same")(x)

autoencoder = keras.Model(inputs=inputs, outputs=outputs)
encoder = keras.Model(inputs=inputs, outputs=latent)

autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
# X shape is (1024, 1)
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, shuffle=True)

# Encode test data
X_encoded = encoder.predict(X_test)

np.save('model_outputs/autoencoder_output.npy', X_encoded)
np.save('model_outputs/autoencoder_labels.npy', y_test)

print("Model output and labels saved.")