# Privacy-Preserving Federated Learning for Cyberattack Detection in Connected Vehicles

A runnable prototype: real data download -> Non-IID split across 3 simulated
vehicles -> centralized baseline -> Federated Learning (Flower) with a basic
Differential Privacy mechanism -> SHAP explainability.

## Files

| File | Phase | What it does |
|---|---|---|
| `common.py` | shared | Data loading + the ONE neural network architecture used everywhere |
| `01_download_preprocess.py` | 1 | Downloads KDD Cup 99, cleans it, Non-IID splits into `data/vehicle_1_data.csv`, `_2_`, `_3_` |
| `02_baseline_centralized.py` | 2 | Trains a normal (non-federated) model on all the data, prints baseline accuracy |
| `03_client.py` | 3 | One simulated vehicle. Run it 3 times (once per vehicle) |
| `04_server.py` | 4 | Flower server: FedAvg + Gaussian-noise Differential Privacy |
| `05_xai_shap.py` | 5 | SHAP explanation of the final model on a real cyberattack sample |

## 1. Setup (once)

Open a terminal in this folder and run:

```bash
python3 -m venv venv                 # create an isolated Python environment
source venv/bin/activate             # on Windows use: venv\Scripts\activate
pip install -r requirements.txt      # install all required libraries
```

## 2. Phase 1 - Download & preprocess the real dataset

```bash
python3 01_download_preprocess.py
```

This creates a `data/` folder containing:
- `preprocessed_full_dataset.csv` (whole dataset, used by steps 2 and 5)
- `vehicle_1_data.csv`, `vehicle_2_data.csv`, `vehicle_3_data.csv` (the 3 Non-IID vehicle shards)
- `scaler.joblib` (the fitted feature scaler)

The script prints how many rows/attacks each vehicle ends up with — you
should see clearly unequal, Non-IID splits.

## 3. Phase 2 - Centralized baseline

```bash
python3 02_baseline_centralized.py
```

Prints the baseline test accuracy and saves `global_model_baseline.h5`.
This is what you compare the federated result against in your paper.

## 4. Phases 3 & 4 - Run the Federated Learning simulation

You need **4 terminals open at the same time**, all in this same folder
(remember to `source venv/bin/activate` in each new terminal first).

**Terminal 1 — start the server first:**
```bash
python3 04_server.py
```
It will print that it's waiting for 3 vehicles to connect.

**Terminal 2:**
```bash
python3 03_client.py --cid 1
```

**Terminal 3:**
```bash
python3 03_client.py --cid 2
```

**Terminal 4:**
```bash
python3 03_client.py --cid 3
```

As soon as all 3 clients have connected, training starts automatically for
3 rounds. Watch Terminal 1 — after the final round it saves
`global_model_federated.h5`, the finished privacy-preserving global model.

> Tuning the privacy/accuracy trade-off: open `04_server.py` and change
> `NOISE_STD_DEV`. Higher = more privacy, lower federated accuracy.
> Lower = closer to the centralized baseline accuracy, less privacy.

## 5. Phase 5 - Explain the model with SHAP

Once `04_server.py` has finished and saved `global_model_federated.h5`:

```bash
python3 05_xai_shap.py
```

This saves two images in this folder:
- `shap_single_attack_explanation.png` — a waterfall plot showing exactly
  which features pushed ONE specific real attack sample's prediction toward
  "Cyberattack" (and by how much).
- `shap_summary_all_attacks.png` — a summary/beeswarm plot showing which
  features matter most for detecting cyberattacks across 30 real attack samples.

## Notes for your paper

- **Dataset**: KDD Cup 99 (10% subset), a real, widely-cited network
  intrusion benchmark, downloaded automatically via
  `sklearn.datasets.fetch_kddcup99`. It stands in for connected-vehicle
  network traffic; swap the loader in `01_download_preprocess.py` for an
  automotive CAN-bus dataset (e.g. Car-Hacking, CICIoV2024) if you need
  domain-specific data later — nothing else in the pipeline needs to change.
- **Non-IID split**: uses a Dirichlet-distribution partition (a standard
  technique in FL research), so each simulated vehicle sees a different,
  realistic mix of attack types.
- **Differential Privacy**: the server adds Gaussian noise to the averaged
  weights each round (a simplified DP-FedAvg). This is enough to demonstrate
  the concept in a prototype, but is not a formally proven (epsilon,
  delta)-DP guarantee — for that, cite/extend with a library such as
  `tensorflow-privacy` or `opacus` and add per-client update clipping.
- **Model**: a small 2-hidden-layer Dense network (32 -> 16 -> 1), chosen so
  each federated round trains in seconds on a laptop CPU.
