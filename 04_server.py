"""
STEP 4 - FEDERATED SERVER + DIFFERENTIAL PRIVACY
----------------------------------------------------
This script plays the role of the neutral, central coordinator (e.g.
an OEM's backend service). It NEVER sees any vehicle's raw data - it
only receives model weight-updates from each vehicle, averages them
together (FedAvg = Federated Averaging), and sends the improved
"global model" back out for the next round.

DIFFERENTIAL PRIVACY (simplified, for demonstration):
Even model weights can leak information about the data used to train
them. To add a basic privacy safeguard, this server adds random
Gaussian ("white") noise to the averaged weights after every round,
before they are sent back to the vehicles. This is the same core idea
used in real DP-FedAvg algorithms (bigger noise = more privacy but
less accuracy). NOTE: this is a simplified, pedagogical version for a
prototype - a production system would also CLIP each client's update
before averaging and use a formally proven DP accountant (e.g. via the
`opacus` or `tensorflow-privacy` libraries) to report an exact
epsilon (privacy budget) instead of just adding raw noise.
"""

import numpy as np                                                       # numpy: used to generate the Gaussian noise
import flwr as fl                                                        # flwr: the Flower federated learning framework
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters   # Helpers to convert between Flower's format and plain numpy arrays
from common import load_dataset, build_model                              # Reuse the shared model architecture

# ----------------------------------------------------------------------
# 0. Settings you can tweak
# ----------------------------------------------------------------------
NUM_ROUNDS = 3                 # How many times the server and vehicles go back and forth
NOISE_STD_DEV = 0.01             # Standard deviation of the Gaussian noise added for Differential Privacy (bigger = more private, less accurate)
MIN_CLIENTS = 3                  # Wait for all 3 simulated vehicles before starting a round

# We need to know the model's input size (number of features) to be able to
# rebuild it and save it to disk once training is finished.
_X, _y = load_dataset("data/preprocessed_full_dataset.csv")
INPUT_DIM = _X.shape[1]


# ----------------------------------------------------------------------
# 1. A custom FedAvg strategy that adds Differential-Privacy noise
# ----------------------------------------------------------------------
class DPFedAvg(fl.server.strategy.FedAvg):
    """
    Identical to normal FedAvg, except that right after averaging all the
    vehicles' weight-updates together, we inject a small amount of random
    Gaussian noise into every single weight value. This is what gives us
    "Differentially Private Federated Averaging" (DP-FedAvg) in its most
    basic form.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters = None   # Will keep a copy of the newest global weights, so we can save them to disk at the end

    def aggregate_fit(self, server_round, results, failures):
        # First, let Flower's built-in FedAvg do the normal weighted averaging
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert Flower's internal Parameters object into a plain list of numpy arrays (one array per model layer)
            weight_arrays = parameters_to_ndarrays(aggregated_parameters)

            # Add independent Gaussian noise (mean 0, std NOISE_STD_DEV) to every weight value in every layer
            noisy_weight_arrays = [
                layer_weights + np.random.normal(loc=0.0, scale=NOISE_STD_DEV, size=layer_weights.shape)
                for layer_weights in weight_arrays
            ]

            print(f"[Server] Round {server_round}: aggregated weights from {len(results)} vehicles "
                  f"and added Differential-Privacy noise (std={NOISE_STD_DEV}).")

            self.latest_parameters = noisy_weight_arrays                          # Remember these for saving to disk later
            aggregated_parameters = ndarrays_to_parameters(noisy_weight_arrays)   # Convert back into Flower's format

        return aggregated_parameters, aggregated_metrics


# ----------------------------------------------------------------------
# 2. Configure and start the Flower server
# ----------------------------------------------------------------------
strategy = DPFedAvg(
    min_fit_clients=MIN_CLIENTS,          # Don't start a training round until this many vehicles have connected
    min_evaluate_clients=MIN_CLIENTS,      # Same, but for the evaluation step
    min_available_clients=MIN_CLIENTS,     # Wait for this many vehicles to be online at all before doing anything
)

if __name__ == "__main__":
    print(f"Starting Flower server for {NUM_ROUNDS} rounds, waiting for {MIN_CLIENTS} vehicles to connect...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",                          # Listen on all network interfaces, port 8080
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),   # Run this many rounds of federated training
        strategy=strategy,
    )

    # ------------------------------------------------------------------
    # 3. Training is finished - rebuild the model and save the FINAL
    #    noisy global weights to disk so the XAI script (step 5) can
    #    load and explain the finished model.
    # ------------------------------------------------------------------
    if strategy.latest_parameters is not None:
        final_model = build_model(input_dim=INPUT_DIM)
        final_model.set_weights(strategy.latest_parameters)
        final_model.save("global_model_federated.h5")
        print("\nSaved final federated + DP global model -> global_model_federated.h5")
    else:
        print("\nWarning: no aggregated parameters were produced - nothing was saved.")
