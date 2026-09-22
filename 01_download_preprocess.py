"""
STEP 1 - DOWNLOAD & PREPROCESS
-------------------------------
This script:
  1. Downloads a REAL, public network-intrusion dataset (KDD Cup 99)
     directly from scikit-learn's built-in dataset loader. No manual
     downloads needed - scikit-learn fetches and caches it for you.
  2. Cleans it and converts the text attack-labels into numbers
     (0 = Normal, 1 = Cyberattack).
  3. Splits the data into 3 unequal, Non-IID chunks, one per
     simulated connected vehicle, to mimic 3 cars that each see a
     different mix of traffic depending on where they are driving
     (e.g. one car mostly sees Normal city traffic, another sees a
     heavy burst of Denial-of-Service-style attacks on the highway).

NOTE ON THE DATASET:
KDD Cup 99 is a classic, widely-used NETWORK intrusion dataset (it
records things like connection duration, bytes sent/received, error
rates, etc). It is not literally CAN-bus / in-vehicle network data,
but it is a standard, easy-to-download stand-in that lets your
prototype demonstrate the full pipeline (download -> Non-IID split ->
Federated Learning -> Differential Privacy -> XAI) with REAL data and
REAL attack patterns (e.g. 'neptune' = a SYN-flood Denial-of-Service
attack - exactly the kind of attack a connected vehicle's telematics
unit could suffer). If your paper later needs an automotive-specific
dataset (e.g. the Car-Hacking / CICIoV2024 CAN-bus datasets), you can
swap the loading code in this ONE file and everything downstream
(baseline, federated clients, server, XAI) keeps working unchanged.
"""

import os                                    # os: lets us create folders on disk
import numpy as np                           # numpy: fast numeric arrays, used for the Non-IID splitting logic
import pandas as pd                          # pandas: turns raw arrays into a labeled table (DataFrame) we can easily manipulate
from sklearn.datasets import fetch_kddcup99  # scikit-learn's built-in downloader for the KDD Cup 99 dataset
from sklearn.preprocessing import StandardScaler  # scales numeric columns so they all have a similar range (helps the neural net learn)
import joblib                                # joblib: used to save the fitted scaler to disk for reuse later

# ----------------------------------------------------------------------
# 0. Settings you can tweak
# ----------------------------------------------------------------------
RANDOM_SEED = 42          # Fixed seed so the download/shuffle/split is reproducible every time you run this script
N_VEHICLES = 3             # We are simulating 3 connected vehicles
NON_IID_ALPHA = 0.3        # Smaller = more unequal (Non-IID) split between vehicles. 0.3 gives a strongly skewed split.
OUTPUT_DIR = "data"        # Folder where all the CSV files will be saved

os.makedirs(OUTPUT_DIR, exist_ok=True)   # Create the "data" folder if it does not already exist

# ----------------------------------------------------------------------
# 1. Download the real dataset
# ----------------------------------------------------------------------
print("Downloading KDD Cup 99 (10% subset) - this may take a minute the first time...")
# percent10=True downloads the smaller ~10% version of the dataset (about 494,000 rows), which
# is plenty of real data for a laptop demo and trains much faster than the full 4.9-million-row version.
kdd = fetch_kddcup99(percent10=True, random_state=RANDOM_SEED, shuffle=True)

# The 41 official KDD Cup 99 column names, in the exact order scikit-learn returns them
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

df = pd.DataFrame(kdd["data"], columns=COLUMN_NAMES)   # Put the raw feature array into a labeled pandas table
raw_labels = kdd["target"]                              # The attack-type labels, e.g. b'normal.', b'neptune.', b'smurf.'

# ----------------------------------------------------------------------
# 2. Clean the text columns
# ----------------------------------------------------------------------
# scikit-learn gives us byte-strings like b'tcp' instead of plain text 'tcp' -
# this loop decodes every byte-string column into a normal Python string.
for col in ["protocol_type", "service", "flag"]:
    df[col] = df[col].apply(lambda v: v.decode("utf-8") if isinstance(v, bytes) else str(v))

# Every other column should be a plain number - force them to float type
# (they arrive as generic "object" dtype because the raw array mixes text and numbers).
numeric_cols = [c for c in COLUMN_NAMES if c not in ["protocol_type", "service", "flag"]]
df[numeric_cols] = df[numeric_cols].astype("float32")

# ----------------------------------------------------------------------
# 3. Turn the attack-type labels into numbers
# ----------------------------------------------------------------------
# Decode the raw labels and strip the trailing "." that KDD Cup 99 always adds, e.g. b'neptune.' -> 'neptune'
label_multi = [l.decode("utf-8").rstrip(".") if isinstance(l, bytes) else str(l).rstrip(".") for l in raw_labels]
df["label_multi"] = label_multi                          # Keep the detailed attack name (used only for the Non-IID split below)
df["label"] = (df["label_multi"] != "normal").astype("int32")  # The label the model will actually learn: 0 = Normal, 1 = Any cyberattack

print(f"Total rows downloaded: {len(df):,}")
print(f"Normal rows: {(df['label'] == 0).sum():,} | Attack rows: {(df['label'] == 1).sum():,}")
print("Attack types present:", sorted(df.loc[df['label'] == 1, 'label_multi'].unique()))

# ----------------------------------------------------------------------
# 4. One-Hot-Encode the 3 text columns and scale the numeric columns
# ----------------------------------------------------------------------
# IMPORTANT: we one-hot-encode the WHOLE dataset BEFORE splitting it between vehicles.
# If we split first and one-hot-encoded each vehicle's chunk separately, a vehicle that
# never happened to see e.g. the "udp" protocol would end up with a DIFFERENT number of
# columns than the other vehicles - which would break Federated Learning, because every
# client's model must have an input layer of the exact same size.
df_encoded = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])

scaler = StandardScaler()   # StandardScaler rescales each numeric column to mean 0, standard deviation 1
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])  # Fit + apply the scaler to the numeric columns only
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))              # Save the fitted scaler to disk in case you need it later

# Save the fully preprocessed, single-table version - this is what the CENTRALIZED
# baseline model (step 2) and the SHAP explainability script (step 5) will use.
full_path = os.path.join(OUTPUT_DIR, "preprocessed_full_dataset.csv")
df_encoded.to_csv(full_path, index=False)
print(f"Saved full preprocessed dataset -> {full_path}  (shape={df_encoded.shape})")

# ----------------------------------------------------------------------
# 5. Split the data into 3 NON-IID chunks, one per simulated vehicle
# ----------------------------------------------------------------------
def non_iid_split(dataframe, label_col, n_clients, alpha, seed):
    """
    Splits `dataframe` into `n_clients` unequal pieces where each client
    gets a different MIX of classes - simulating vehicles that experience
    different driving environments (e.g. one car mostly sees Normal
    traffic on a quiet suburban route, another sees a big burst of
    DoS-style attacks while parked at a busy public charging station).

    How it works: for EACH class label separately, we draw a random
    proportion for each client from a Dirichlet distribution (a standard
    technique for creating Non-IID splits in Federated Learning research).
    A small `alpha` (e.g. 0.3) produces a very skewed / unequal split;
    a large `alpha` (e.g. 100) would produce an almost even split.
    """
    rng = np.random.default_rng(seed)                        # A reproducible random number generator
    client_indices = [[] for _ in range(n_clients)]           # One empty list of row-indices per client, to be filled in below

    for class_value in dataframe[label_col].unique():         # Loop over every distinct attack-type label
        class_idx = dataframe.index[dataframe[label_col] == class_value].to_numpy()  # All row indices belonging to this one class
        rng.shuffle(class_idx)                                # Shuffle them so the split isn't influenced by their original order

        proportions = rng.dirichlet(alpha=[alpha] * n_clients)          # e.g. [0.71, 0.05, 0.24] - how much of this class each client gets
        split_points = (np.cumsum(proportions) * len(class_idx)).astype(int)[:-1]  # Turn proportions into actual cut positions
        chunks = np.split(class_idx, split_points)             # Cut the shuffled indices into n_clients unequal pieces

        for client_id, chunk in enumerate(chunks):             # Hand each piece to its client
            client_indices[client_id].extend(chunk.tolist())

    return client_indices


client_row_indices = non_iid_split(
    df_encoded, label_col="label_multi", n_clients=N_VEHICLES, alpha=NON_IID_ALPHA, seed=RANDOM_SEED
)

print("\nNon-IID split summary (rows per vehicle, and % that are attacks):")
for i, idx in enumerate(client_row_indices, start=1):
    vehicle_df = df_encoded.loc[idx].drop(columns=["label_multi"])   # Drop the detailed label - the model only needs the binary one
    save_path = os.path.join(OUTPUT_DIR, f"vehicle_{i}_data.csv")
    vehicle_df.to_csv(save_path, index=False)                        # Save this vehicle's private local data to its own CSV file
    attack_pct = 100 * vehicle_df["label"].mean()
    print(f"  vehicle_{i}_data.csv -> {len(vehicle_df):>6,} rows | {attack_pct:5.1f}% are cyberattacks")

print("\nDone! Every vehicle's data never leaves its own CSV file - "
      "that separation is what makes the setup 'Federated' in step 3/4.")
