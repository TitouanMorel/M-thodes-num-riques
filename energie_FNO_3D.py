import numpy as np
import torch
from neuralop.losses import LpLoss
from codecarbon import EmissionsTracker
import csv
import os

# Paramètres physiques
D, T, nx, K, border = 0.01, 0.1, 32, 8, 0.0
TRAIN_CSV = "fno_energie_entrainement_3D.csv"
RESULTS_FILE = "fno_energie_inference_3D.csv"
N_SAMPLES = 100
N_INFERENCES = 100


def dataset(N, D, T, nx, K=8, border=0.0):
    x = np.linspace(0, 1, nx, endpoint=False)
    k_idx = np.arange(1, K + 1)
    sin_x = np.sin(np.outer(k_idx, np.pi * x))  # (K, nx)
    k2 = k_idx[:, None, None] ** 2
    l2 = k_idx[None, :, None] ** 2
    m2 = k_idx[None, None, :] ** 2
    atten = np.exp(-D * (k2 + l2 + m2) * np.pi**2 * T)  # (K, K, K)

    u0s, uTs = [], []
    for _ in range(N):
        c = np.random.uniform(-20, 20, size=(K, K, K))
        u0 = border + np.einsum("klm,kx,ly,mz->xyz", c, sin_x, sin_x, sin_x)
        uT = border + np.einsum("klm,kx,ly,mz->xyz", c * atten, sin_x, sin_x, sin_x)
        u0s.append(u0.astype(np.float32))
        uTs.append(uT.astype(np.float32))
    return np.array(u0s), np.array(uTs)


print("Génération des données de test 3D...", flush=True)
u0s, uTs = dataset(N_SAMPLES, D, T, nx, K, border)
X_test = torch.tensor(u0s).unsqueeze(1)

Y_test = torch.tensor(uTs).unsqueeze(1)

with open(TRAIN_CSV) as f:
    models = [(row["config"], row["model_path"]) for row in csv.DictReader(f)]
print(f"{len(models)} architectures à tester", flush=True)

loss_fn = LpLoss(d=3, p=2)
file_exists = os.path.isfile(RESULTS_FILE)

with open(RESULTS_FILE, "a", newline="") as fout:
    writer = csv.writer(fout)
    if not file_exists:
        writer.writerow(
            ["config", "model_path", "energy_per_inference_joules", "mean_l2_error"]
        )

    for config, model_path in models:
        if not os.path.isfile(model_path):
            print(f"[SKIP] {model_path} introuvable", flush=True)
            continue

        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()

        tracker = EmissionsTracker(
            save_to_file=False, measure_power_secs=1, log_level="error"
        )
        tracker.start()
        preds = []
        with torch.no_grad():
            for i in range(N_INFERENCES):
                preds.append(model(X_test[i : i + 1]))
        tracker.stop()
        energy = tracker.final_emissions_data.energy_consumed * 3.6e6 / N_INFERENCES

        errors = [
            loss_fn(preds[i], Y_test[i : i + 1]).item() for i in range(N_INFERENCES)
        ]
        mean_error = float(np.mean(errors))

        print(f"Config {config} : {energy:.4e} J/inf | L²={mean_error:.4e}", flush=True)
        writer.writerow([config, model_path, energy, mean_error])
        fout.flush()

print(f"Enregistré dans {RESULTS_FILE}", flush=True)
