import argparse
import numpy as np
import torch
from neuralop.losses import LpLoss
from codecarbon import EmissionsTracker
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("--config", type=str, default="")
args = parser.parse_args()

RESULTS_FILE = "fno_energie_results_L_T01.csv"
N_INFERENCES = 100

print("Chargement dataset_Lshape_T01.npz...", flush=True)
data = np.load("dataset_Lshape_T01.npz")
u0s  = data["u0s"]
uTs  = data["uTs"]
mask = data["mask"]

X_test = torch.tensor(u0s[800:], dtype=torch.float32).unsqueeze(1)
Y_test = torch.tensor(uTs[800:], dtype=torch.float32).unsqueeze(1)
mask_t = torch.tensor(mask, dtype=torch.float32)

print(f"Chargement : {args.model_path}", flush=True)
loaded = torch.load(args.model_path, map_location="cpu", weights_only=False)
model = loaded if not isinstance(loaded, dict) else None
if model is None:
    raise ValueError("state_dict détecté")
model.eval()

base_loss = LpLoss(d=2, p=2)
tracker = EmissionsTracker(save_to_file=False, measure_power_secs=1, log_level="error")
tracker.start()

preds = []
with torch.no_grad():
    for i in range(N_INFERENCES):
        preds.append(model(X_test[i % len(X_test) : i % len(X_test) + 1]))

tracker.stop()
energy_per_inference = tracker.final_emissions_data.energy_consumed * 3.6e6 / N_INFERENCES

errors = [
    base_loss(preds[i] * mask_t, Y_test[i % len(X_test) : i % len(X_test) + 1] * mask_t).item()
    for i in range(N_INFERENCES)
]
mean_error = float(np.mean(errors))

print(f"Énergie par inférence : {energy_per_inference:.4e} J", flush=True)
print(f"Erreur L² masquée     : {mean_error:.4e}", flush=True)

file_exists = os.path.isfile(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["config", "model_path", "energy_per_inference_joules", "mean_l2_error_masked"])
    writer.writerow([args.config, args.model_path, energy_per_inference, mean_error])

print(f"Enregistré dans {RESULTS_FILE}", flush=True)
