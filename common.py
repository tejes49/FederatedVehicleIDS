"""
common.py
---------
Shared helper functions used by every script in this project.
Keeping the model definition and data-loading logic in ONE place
guarantees that the baseline model, every federated "vehicle" client,
and the federated server all build the EXACT SAME neural network
architecture. This is critical for Federated Learning: all the
clients and the server must agree on the shape of the model weights,
otherwise averaging the weights together would crash or produce
garbage.
"""

import pandas as pd                     # pandas: used to read our CSV files into tables (DataFrames)
import tensorflow as tf                 # tensorflow: the deep learning library we use to build the neural network


def load_dataset(csv_path):
    """
    Reads one CSV file and splits it into:
      X -> the input features (network connection statistics)
      y -> the label (0 = Normal traffic, 1 = Cyberattack)

    Any column called 'label_multi' (the original, detailed attack name,
    e.g. 'neptune', 'smurf', 'normal') is only used earlier to create the
    Non-IID splits, so we drop it here - the model only ever sees the
    simple binary 'label' column.
    """
    df = pd.read_csv(csv_path)                      # Load the CSV file from disk into a pandas table
    if "label_multi" in df.columns:                 # Check if the detailed attack-type column is present
        df = df.drop(columns=["label_multi"])        # If so, remove it - the model does not need it
    y = df["label"].values.astype("float32")         # Pull out the binary label column as our target (0 or 1)
    X = df.drop(columns=["label"]).values.astype("float32")  # Everything else is our input features
    return X, y                                       # Give both back to whoever called this function


def build_model(input_dim):
    """
    Builds a small, fast neural network that acts as our
    "Intrusion Detection System" (IDS). It is intentionally simple
    (only a few Dense/fully-connected layers) so that it trains in
    seconds on a normal laptop, even inside a Federated Learning loop
    where it has to be retrained many times.
    """
    model = tf.keras.Sequential([                                    # Sequential = a simple stack of layers, one after another
        tf.keras.layers.Input(shape=(input_dim,)),                    # The input layer: expects one row of `input_dim` numeric features
        tf.keras.layers.Dense(32, activation="relu"),                 # Hidden layer 1: 32 neurons, ReLU activation (learns non-linear patterns)
        tf.keras.layers.Dense(16, activation="relu"),                 # Hidden layer 2: 16 neurons, ReLU activation (learns higher-level patterns)
        tf.keras.layers.Dense(1, activation="sigmoid"),                # Output layer: 1 neuron, sigmoid squashes the output into a 0-1 probability
    ])
    model.compile(
        optimizer="adam",                 # Adam: a standard, reliable optimizer that adjusts the network's weights during training
        loss="binary_crossentropy",       # Binary cross-entropy: the standard loss function for a 2-class (Normal vs Attack) problem
        metrics=["accuracy"],             # Track accuracy (percentage of correct predictions) while training
    )
    return model
