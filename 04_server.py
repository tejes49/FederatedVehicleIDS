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
less accuracy).

NOTE: this is a simplified, pedagogical version for a prototype.
A production system would also CLIP each client's update before
averaging and use a formally proven DP accountant (e.g. via the
opacus or tensorflow-privacy libraries) to report an exact epsilon
(privacy budget) instead of just adding raw noise.
"""

import numpy as np
import flwr as fl

from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from common import load_dataset, build_model


# ----------------------------------------------------------------------
# 0. Settings you can tweak
# ----------------------------------------------------------------------

NUM_ROUNDS = 3
NOISE_STD_DEV = 0.01
MIN_CLIENTS = 3

# Flower server port
SERVER_PORT = 8081


# ----------------------------------------------------------------------
# We need to know the model's input size
# ----------------------------------------------------------------------

_X, _y = load_dataset(
    "data/preprocessed_full_dataset.csv"
)

INPUT_DIM = _X.shape[1]


# ----------------------------------------------------------------------
# 1. A custom FedAvg strategy that adds Differential-Privacy noise
# ----------------------------------------------------------------------

class DPFedAvg(fl.server.strategy.FedAvg):

    """
    Identical to normal FedAvg, except that right after averaging
    all the vehicles' weight-updates together, we inject a small
    amount of random Gaussian noise into every single weight value.

    This gives us a basic demonstration of Differentially Private
    Federated Averaging (DP-FedAvg).
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Keep a copy of the newest global weights
        # so that we can save them after training.
        self.latest_parameters = None


    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        # --------------------------------------------------------------
        # First perform normal FedAvg
        # --------------------------------------------------------------

        aggregated_parameters, aggregated_metrics = (
            super().aggregate_fit(
                server_round,
                results,
                failures
            )
        )


        # --------------------------------------------------------------
        # Add Differential Privacy noise
        # --------------------------------------------------------------

        if aggregated_parameters is not None:

            # Convert Flower Parameters into NumPy arrays
            weight_arrays = parameters_to_ndarrays(
                aggregated_parameters
            )


            # Add Gaussian noise to every model weight
            noisy_weight_arrays = [

                layer_weights
                + np.random.normal(
                    loc=0.0,
                    scale=NOISE_STD_DEV,
                    size=layer_weights.shape
                )

                for layer_weights in weight_arrays
            ]


            print(
                f"[Server] Round {server_round}: "
                f"aggregated weights from {len(results)} vehicles "
                f"and added Differential-Privacy noise "
                f"(std={NOISE_STD_DEV})."
            )


            # Store the latest noisy global model
            self.latest_parameters = noisy_weight_arrays


            # Convert NumPy arrays back into Flower Parameters
            aggregated_parameters = ndarrays_to_parameters(
                noisy_weight_arrays
            )


        return aggregated_parameters, aggregated_metrics


# ----------------------------------------------------------------------
# 2. Configure the Flower server
# ----------------------------------------------------------------------

strategy = DPFedAvg(

    # Don't start a training round until 3 vehicles connect
    min_fit_clients=MIN_CLIENTS,

    # Wait for 3 vehicles during evaluation
    min_evaluate_clients=MIN_CLIENTS,

    # Require 3 vehicles to be available
    min_available_clients=MIN_CLIENTS,
)


# ----------------------------------------------------------------------
# 3. Start the Flower server
# ----------------------------------------------------------------------

if __name__ == "__main__":

    print(
        f"Starting Flower server for {NUM_ROUNDS} rounds, "
        f"waiting for {MIN_CLIENTS} vehicles to connect..."
    )

    print(
        f"Flower server listening on port {SERVER_PORT}..."
    )


    fl.server.start_server(

        # --------------------------------------------------------------
        # CHANGED FROM 8080 TO 8081
        # --------------------------------------------------------------

        server_address=f"0.0.0.0:{SERVER_PORT}",

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy,
    )


    # ------------------------------------------------------------------
    # 4. Training is finished
    # ------------------------------------------------------------------

    # Rebuild the model and save the final noisy global weights
    # so that Step 5 (XAI) can load and explain the finished model.

    if strategy.latest_parameters is not None:

        final_model = build_model(
            input_dim=INPUT_DIM
        )


        final_model.set_weights(
            strategy.latest_parameters
        )


        final_model.save(
            "global_model_federated.h5"
        )


        print(
            "\nSaved final federated + DP global model "
            "-> global_model_federated.h5"
        )

    else:

        print(
            "\nWarning: no aggregated parameters were produced "
            "- nothing was saved."
        )
