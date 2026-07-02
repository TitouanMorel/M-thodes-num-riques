import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from neuralop.models import FNO
from neuralop.training import Trainer
from neuralop.losses import LpLoss
from codecarbon import EmissionsTracker
import csv
import os
import fcntl

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config",   type=str, required=True)
parser.add_argument("--n_modes",  type=int, required=True)
parser.add_argument("--hidden",   type=int, required=True)
parser.add_argument("--n_layers", type=int, required=True)
parser.add_argument("--n_epochs", type=int, default=500)
args = parser.parse_args()

CONFIG   = args.config
N_MODES  = args.n_modes
HIDDEN   = args.hidden
N_LAYERS = args.n_layers
N_EPOCHS = args.n_epochs

MODEL_PATH   = f"fno_config{CONFIG}_{N_EPOCHS}epoch_.pth"
RESULTS_FILE = "fno_energie_entrainement.csv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset(N, D, T, dx, border=0.0):
    x = np.arange(0, 1, dx)
    y = np.arange(0, 1, dx)
    X_grid, Y_grid = np.meshgrid(x, y)
    K = 16
    u0s, uTs = [], []
    for _ in range(N):
        c = np.random.uniform(-20, 20, size=(K, K))
        u0 = border + sum(
            c[k, l] * np.sin((k+1)*np.pi*X_grid) * np.sin((l+1)*np.pi*Y_grid)
            for k in range(K) for l in range(K)
        )
        uT = border + sum(
            c[k, l]
            * np.exp(-D * ((k+1)**2 + (l+1)**2) * np.pi**2 * T)
            * np.sin((k+1)*np.pi*X_grid) * np.sin((l+1)*np.pi*Y_grid)
            for k in range(K) for l in range(K)
        )
        u0s.append(u0)
        uTs.append(uT)
    return np.array(u0s), np.array(uTs)

class DiffusionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {"x": self.X[idx].to(device), "y": self.Y[idx].to(device)}

# Données
N, D, T, dx = 1000, 0.01, 0.1, 1/64
print("Génération des données...", flush=True)
u0s, uTs = dataset(N, D, T, dx)

X = torch.tensor(u0s, dtype=torch.float32).unsqueeze(1)
Y = torch.tensor(uTs, dtype=torch.float32).unsqueeze(1)

train_dataset = DiffusionDataset(X[:800], Y[:800])
test_dataset  = DiffusionDataset(X[800:], Y[800:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loaders = {"test": DataLoader(test_dataset, batch_size=32, shuffle=False)}

# Modèle
model = FNO(
    n_modes=(N_MODES, N_MODES),
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN,
    n_layers=N_LAYERS,
).to(device)

# Optimisation
optimizer   = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler   = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
train_loss  = LpLoss(d=2, p=2)
eval_losses = {"L2": LpLoss(d=2, p=2)}

print(f"Config {CONFIG} : n_modes={N_MODES}, hidden={HIDDEN}, n_layers={N_LAYERS}", flush=True)

trainer = Trainer(
    model=model,
    n_epochs=N_EPOCHS,
    device=str(device),
    verbose=True,
)

# Mesure énergie pendant entraînement
tracker = EmissionsTracker(
    save_to_file=False,
    measure_power_secs=60,
    log_level="error",
    tracking_mode="machine",
)
tracker.start()

trainer.train(
    train_loader=train_loader,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_loss,
    eval_losses=eval_losses,
)

tracker.stop()
energy_train_joules = tracker.final_emissions_data.energy_consumed * 3.6e6
print(f"Énergie entraînement : {energy_train_joules:.4e} J", flush=True)

# Sauvegarde CSV énergie
file_exists = os.path.isfile(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["config", "model_path", "n_modes", "hidden_channels",
                         "n_layers", "n_epochs", "energy_train_joules"])
    writer.writerow([CONFIG, MODEL_PATH, N_MODES, HIDDEN, N_LAYERS,
                     N_EPOCHS, energy_train_joules])
    fcntl.flock(f, fcntl.LOCK_UN)

print(f"Résultats enregistrés dans {RESULTS_FILE}", flush=True)
torch.save(model, MODEL_PATH)
print(f"Modèle sauvegardé : {MODEL_PATH}", flush=True)