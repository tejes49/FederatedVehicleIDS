"""
STEP 5 - EXPLAINABLE AI (XAI) WITH SHAP
------------------------------------------
Once the federated model is trained, it is just a "black box" that
outputs Normal/Attack probabilities. For a security system, that is
not good enough - an analyst needs to know WHY the model raised an
alarm. This script uses SHAP (SHapley Additive exPlanations) to show
exactly which network features pushed one specific real cyberattack
sample's prediction towards "Attack".
"""

import os
import numpy as np
import pandas as pd
import shap                                    # shap: the explainability library
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")                            # Use a non-interactive backend so plots save straight to files, no display window needed
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Load the trained global model
# ----------------------------------------------------------------------
# Prefer the final Federated + Differential-Privacy model; fall back to
# the centralized baseline if you have not run steps 3/4 yet.
if os.path.exists("global_model_federated.h5"):
    model_path = "global_model_federated.h5"
else:
    model_path = "global_model_baseline.h5"
    print("global_model_federated.h5 not found - explaining the baseline model instead.")

model = tf.keras.models.load_model(model_path)
print(f"Loaded model from {model_path}")

# ----------------------------------------------------------------------
# 2. Load the same preprocessed data used everywhere else
# ----------------------------------------------------------------------
df = pd.read_csv("data/preprocessed_full_dataset.csv")
feature_names = [c for c in df.columns if c not in ("label", "label_multi")]  # Every column except the labels is a model input feature
X = df[feature_names].values.astype("float32")
y = df["label"].values

# ----------------------------------------------------------------------
# 3. Pick a small "background" sample (SHAP needs this as a reference
#    point for "what is normal", to measure how much each feature moves
#    the prediction away from that baseline) and one real attack row to explain
# ----------------------------------------------------------------------
rng = np.random.default_rng(42)
background_idx = rng.choice(len(X), size=100, replace=False)   # 100 random rows used as SHAP's reference/background distribution
background = X[background_idx]

attack_indices = np.where(y == 1)[0]                             # Every row index that is a real cyberattack
sample_idx = int(rng.choice(attack_indices))                     # Pick ONE real cyberattack sample to explain
sample = X[sample_idx: sample_idx + 1]                            # Keep it as a (1, num_features) 2-D array, the shape Keras expects

predicted_prob = float(model.predict(sample, verbose=0)[0][0])
print(f"\nExplaining row #{sample_idx} (true label = Cyberattack). "
      f"Model's predicted attack probability: {predicted_prob:.4f}")

# ----------------------------------------------------------------------
# 4. Run SHAP's KernelExplainer
#    (model-agnostic - works with any model.predict() function, so it
#    is compatible with any TensorFlow/Keras version)
# ----------------------------------------------------------------------
predict_fn = lambda data: model.predict(data, verbose=0)    # Wrap the model's predict function so SHAP can call it repeatedly
explainer = shap.KernelExplainer(predict_fn, background)     # Build the explainer using our 100-row background sample

print("Computing SHAP values for the single cyberattack sample (this can take ~30-60 seconds)...")
shap_values_single = explainer.shap_values(sample, nsamples=200)   # nsamples: how many perturbed samples SHAP tests internally - 200 keeps it fast

# shap_values_single comes back as a list (one entry per model output); we have 1 output neuron, so take index 0
sv = np.array(shap_values_single[0]).reshape(-1)   # Flatten to a simple 1-D array: one SHAP value per feature, for our one sample

# ----------------------------------------------------------------------
# 5. Plot: which features pushed the prediction towards "Cyberattack"?
# ----------------------------------------------------------------------
explanation = shap.Explanation(
    values=sv,                                      # The SHAP value (impact) of each feature for this one sample
    base_values=explainer.expected_value[0],         # The model's average output over the background data ("starting point")
    data=sample[0],                                  # The actual feature values of the sample being explained
    feature_names=feature_names,                     # So the plot shows real column names instead of "feature 0, feature 1..."
)

plt.figure()
shap.plots.waterfall(explanation, max_display=15, show=False)   # A waterfall plot: shows the top 15 features that moved the prediction, in order of importance
plt.title(f"Why row #{sample_idx} was flagged as a Cyberattack")
plt.tight_layout()
plt.savefig("shap_single_attack_explanation.png", dpi=150)
print("Saved -> shap_single_attack_explanation.png")

# ----------------------------------------------------------------------
# 6. (Bonus) A broader SHAP summary plot across several attack samples,
#    to see which features matter for cyberattack detection IN GENERAL,
#    not just for this one row.
# ----------------------------------------------------------------------
sample_batch_idx = rng.choice(attack_indices, size=min(30, len(attack_indices)), replace=False)
sample_batch = X[sample_batch_idx]

print("Computing SHAP values for 30 attack samples for the summary plot "
      "(this can take a couple of minutes)...")
shap_values_batch = explainer.shap_values(sample_batch, nsamples=100)
sv_batch = np.array(shap_values_batch[0])   # shape: (30, num_features)

plt.figure()
shap.summary_plot(sv_batch, sample_batch, feature_names=feature_names, show=False, max_display=15)
plt.tight_layout()
plt.savefig("shap_summary_all_attacks.png", dpi=150)
print("Saved -> shap_summary_all_attacks.png")
