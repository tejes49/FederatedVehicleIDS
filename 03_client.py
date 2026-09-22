"""
STEP 3 - FEDERATED CLIENT (represents ONE connected vehicle)
--------------------------------------------------------------
This script represents a single car's on-board computer. It:
  1. Loads ONLY that one vehicle's private local CSV file (its raw
     data NEVER leaves this script).
  2. Trains the shared model architecture on that local data for a
     few epochs whenever the Flower server asks it to.
  3. Sends back only the UPDATED MODEL WEIGHTS (a bunch of numbers),
     never the raw network-traffic rows themselves. That is the core
     privacy idea behind Federated Learning.

Run one of these per simulated vehicle, e.g. in 3 separate terminals:
    python 03_client.py --cid 1
    python 03_client.py --cid 2
    python 03_client.py --cid 3
"""

import argparse                                    # argparse: lets us pass --cid 1 / --cid 2 / --cid 3 on the command line
from sklearn.model_selection import train_test_split
import flwr as fl                                   # flwr: the Flower federated learning framework
from common import load_dataset, build_model          # Reuse the SAME data-loading & model-building code as every other script

# ----------------------------------------------------------------------
# 1. Read which vehicle this is from the command line
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cid", type=int, required=True, help="Vehicle / client ID: 1, 2, or 3")
parser.add_argument("--server", type=str, default="127.0.0.1:8080", help="Address of the Flower server")
args = parser.parse_args()

# ----------------------------------------------------------------------
# 2. Load ONLY this vehicle's own local data (never shared with anyone)
# ----------------------------------------------------------------------
data_path = f"data/vehicle_{args.cid}_data.csv"
X, y = load_dataset(data_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[Vehicle {args.cid}] Loaded {len(X)} local rows ({data_path}); "
      f"train={len(X_train)}, test={len(X_test)}")

# ----------------------------------------------------------------------
# 3. Build this vehicle's local copy of the shared model architecture
# ----------------------------------------------------------------------
model = build_model(input_dim=X_train.shape[1])


# ----------------------------------------------------------------------
# 4. Define the Flower client: this is the "contract" Flower uses to
#    talk to your model (get weights / set weights / train / test).
# ----------------------------------------------------------------------
class VehicleClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        # The server calls this to ask: "what are your current model weights?"
        return model.get_weights()

    def fit(self, parameters, config):
        # The server calls this to say: "here are the latest GLOBAL weights,
        # please train them further using your own local data."
        model.set_weights(parameters)                                    # Load the global weights into this vehicle's local model
        model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=0)   # Train for 1 local epoch on this vehicle's private data
        # Send back: the newly updated local weights, how many rows we trained on
        # (Flower uses this count to weight the averaging fairly), and an empty metrics dict.
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        # The server calls this to check how good the CURRENT global model is
        # on this vehicle's own local test data.
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"[Vehicle {args.cid}] local eval -> loss={loss:.4f}, accuracy={accuracy:.4f}")
        return loss, len(X_test), {"accuracy": accuracy}


# ----------------------------------------------------------------------
# 5. Connect to the Flower server and start participating in training
# ----------------------------------------------------------------------
if __name__ == "__main__":
    fl.client.start_client(
        server_address=args.server,           # Where the server (04_server.py) is listening
        client=VehicleClient().to_client(),   # Wrap our NumPyClient so Flower's core engine can use it
    )
