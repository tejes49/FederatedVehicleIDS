"""
STEP 3 - FEDERATED CLIENT (represents ONE connected vehicle)
--------------------------------------------------------------
This script represents a single car's on-board computer.

It:
  1. Loads ONLY that vehicle's private local CSV file.
  2. Trains the shared model on its local data.
  3. Sends ONLY updated model weights to the Flower server.
  4. Receives the updated global model from the server.

Run one client per simulated vehicle:
    python 03_client.py --cid 1
    python 03_client.py --cid 2
    python 03_client.py --cid 3
"""

import argparse

from sklearn.model_selection import train_test_split
import flwr as fl

from common import load_dataset, build_model


# ----------------------------------------------------------------------
# 1. Read vehicle ID and server address
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Vehicle / client ID: 1, 2, or 3"
)

# IMPORTANT:
# Server was changed from port 8080 to 8081
parser.add_argument(
    "--server",
    type=str,
    default="127.0.0.1:8081",
    help="Address of the Flower server"
)

args = parser.parse_args()


# ----------------------------------------------------------------------
# 2. Load ONLY this vehicle's local data
# ----------------------------------------------------------------------

data_path = f"data/vehicle_{args.cid}_data.csv"

X, y = load_dataset(data_path)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(
    f"[Vehicle {args.cid}] Loaded {len(X)} local rows "
    f"({data_path}); "
    f"train={len(X_train)}, "
    f"test={len(X_test)}"
)


# ----------------------------------------------------------------------
# 3. Build the local copy of the shared model
# ----------------------------------------------------------------------

model = build_model(
    input_dim=X_train.shape[1]
)


# ----------------------------------------------------------------------
# 4. Define Flower client
# ----------------------------------------------------------------------

class VehicleClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        """
        Server asks the vehicle for its current model weights.
        """

        return model.get_weights()


    def fit(self, parameters, config):
        """
        Server sends global weights.

        Vehicle:
        1. Loads global weights.
        2. Trains on private local data.
        3. Sends updated weights back.
        """

        model.set_weights(parameters)

        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=32,
            verbose=0
        )

        print(
            f"[Vehicle {args.cid}] "
            f"Completed local training."
        )

        return (
            model.get_weights(),
            len(X_train),
            {}
        )


    def evaluate(self, parameters, config):
        """
        Evaluate the global model on this vehicle's
        private test data.
        """

        model.set_weights(parameters)

        loss, accuracy = model.evaluate(
            X_test,
            y_test,
            verbose=0
        )

        print(
            f"[Vehicle {args.cid}] local eval -> "
            f"loss={loss:.4f}, "
            f"accuracy={accuracy:.4f}"
        )

        return (
            loss,
            len(X_test),
            {"accuracy": accuracy}
        )


# ----------------------------------------------------------------------
# 5. Connect to Flower server
# ----------------------------------------------------------------------

if __name__ == "__main__":

    print(
        f"[Vehicle {args.cid}] "
        f"Connecting to Flower server at {args.server}..."
    )

    fl.client.start_client(
        server_address=args.server,
        client=VehicleClient().to_client(),
    )
