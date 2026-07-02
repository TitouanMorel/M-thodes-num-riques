import argparse
import numpy as np
import torch
from neuralop.models import FNO
from neuralop.losses import LpLoss
from codecarbon import EmissionsTracker
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("--config", type=str, default="")
parser.add_argument(
    "--n_modes",
    type=int,
    default=None,
    help="Requis seulement si le .pth est un state_dict",
)
parser.add_argument("--hidden", type=int, default=None)
parser.add_argument("--n_layers", type=int, default=None)
args = parser.parse_args()
model_path = args.model_path

# parametres
D, T, dx = 0.01, 0.1, 1 / 64
RESULTS_FILE = "fno_energie_results_T01.csv"


N_SAMPLES = 100
N_INFERENCES = 100
x = np.arange(0, 1, dx)
X_grid, Y_grid = np.meshgrid(x, x)
K, border = 16, 0.0
u0s, uTs = [], []
for _ in range(N_SAMPLES):
    c = np.random.uniform(-20, 20, size=(K, K))
    u0 = border + sum(
        c[k, l] * np.sin((k + 1) * np.pi * X_grid) * np.sin((l + 1) * np.pi * Y_grid)
        for k in range(K)
        for l in range(K)
    )
    uT = border + sum(
        c[k, l]
        * np.exp(-D * ((k + 1) ** 2 + (l + 1) ** 2) * np.pi**2 * T)
        * np.sin((k + 1) * np.pi * X_grid)
        * np.sin((l + 1) * np.pi * Y_grid)
        for k in range(K)
        for l in range(K)
    )
    u0s.append(u0)
    uTs.append(uT)

X_test = torch.tensor(np.array(u0s), dtype=torch.float32).unsqueeze(1)
Y_test = torch.tensor(np.array(uTs), dtype=torch.float32).unsqueeze(1)

print(f"Chargement du modèle : {model_path}", flush=True)
loaded = torch.load(model_path, map_location="cpu", weights_only=False)

if isinstance(loaded, dict):
    if args.n_modes is None or args.hidden is None or args.n_layers is None:
        raise ValueError(
            f"{model_path} est un state_dict : --n_modes, --hidden et --n_layers "
            "sont obligatoires pour reconstruire l'architecture."
        )
    model = FNO(
        n_modes=(args.n_modes, args.n_modes),
        in_channels=1,
        out_channels=1,
        hidden_channels=args.hidden,
        n_layers=args.n_layers,
    )
    model.load_state_dict(loaded)
else:
    model = loaded

model.eval()

loss_fn = LpLoss(d=2, p=2)
tracker = EmissionsTracker(save_to_file=False, measure_power_secs=1, log_level="error")
tracker.start()

preds = []
with torch.no_grad():
    for i in range(N_INFERENCES):
        preds.append(model(X_test[i : i + 1]))

tracker.stop()
energy_per_inference = (
    tracker.final_emissions_data.energy_consumed * 3.6e6 / N_INFERENCES
)
errors = [loss_fn(preds[i], Y_test[i : i + 1]).item() for i in range(N_SAMPLES)]
mean_error = float(np.mean(errors))

print(f"Énergie par inférence : {energy_per_inference:.4e} J", flush=True)
print(f"Erreur L² moyenne     : {mean_error:.4e}", flush=True)


file_exists = os.path.isfile(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(
            ["config", "model_path", "energy_per_inference_joules", "mean_l2_error"]
        )
    writer.writerow([args.config, model_path, energy_per_inference, mean_error])

print(f"Enregistré dans {RESULTS_FILE}", flush=True)
