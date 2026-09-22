"""
STEP 2 - CENTRALIZED BASELINE MODEL
------------------------------------
This trains ONE ordinary (non-federated) neural network on ALL of the
data at once, exactly like a traditional "send all data to one central
server" approach. This is our point of COMPARISON: later, we want to
check that the Federated + Differential-Privacy model (steps 3 & 4) is
still reasonably close in accuracy to this centralized baseline, even
though no vehicle ever shared its raw data.
"""

from sklearn.model_selection import train_test_split   # Splits data into a training set and a held-out test set
from common import load_dataset, build_model             # Reuse our shared data-loading and model-building functions

# ----------------------------------------------------------------------
# 1. Load the full, preprocessed dataset created in step 1
# ----------------------------------------------------------------------
X, y = load_dataset("data/preprocessed_full_dataset.csv")   # X = features, y = 0/1 attack label
print(f"Loaded {X.shape[0]:,} rows with {X.shape[1]} features each.")

# ----------------------------------------------------------------------
# 2. Split into train (80%) and test (20%) sets
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # Hold back 20% of the data purely for evaluating accuracy
    random_state=42,      # Reproducible split
    stratify=y,            # Keep the same Normal/Attack ratio in both the train and test sets
)

# ----------------------------------------------------------------------
# 3. Build and train the model
# ----------------------------------------------------------------------
model = build_model(input_dim=X_train.shape[1])   # Build the same small neural network architecture used everywhere in this project

print("\nTraining the centralized baseline model...")
model.fit(
    X_train, y_train,
    validation_split=0.1,   # Use 10% of the training data to monitor for overfitting during training
    epochs=5,                # 5 passes over the training data - enough for this simple model to converge quickly
    batch_size=256,          # Process 256 rows at a time - a good speed/stability trade-off on a laptop CPU
    verbose=2,                # Print one line of progress per epoch (keeps the output short and readable)
)

# ----------------------------------------------------------------------
# 4. Evaluate on the untouched test set
# ----------------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n=== CENTRALIZED BASELINE TEST ACCURACY: {test_accuracy * 100:.2f}% ===")

# ----------------------------------------------------------------------
# 5. Save the trained model so you can compare it side-by-side with
#    the federated model later if you want to.
# ----------------------------------------------------------------------
model.save("global_model_baseline.h5")
print("Saved trained baseline model -> global_model_baseline.h5")
