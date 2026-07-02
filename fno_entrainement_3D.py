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

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--n_modes", type=int, required=True)
parser.add_argument("--hidden", type=int, required=True)
parser.add_argument("--n_layers", type=int, required=True)
parser.add_argument("--n_epochs", type=int, default=500)
args = parser.parse_args()

CONFIG = args.config
N_MODES = args.n_modes
HIDDEN = args.hidden
N_LAYERS = args.n_layers
N_EPOCHS = args.n_epochs

MODEL_PATH = f"fno3D_config{CONFIG}_{N_EPOCHS}epoch.pth"
RESULTS_FILE = "fno_energie_entrainement_3D.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}", flush=True)


def dataset(N, D, T, nx, K=8, border=0.0):

    x = np.linspace(0, 1, nx, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

    k_idx = np.arange(1, K + 1)
    sin_x = np.sin(np.outer(k_idx, np.pi * x))  # (K, nx)
    sin_y = sin_x.copy()
    sin_z = sin_x.copy()

    k2 = k_idx[:, None, None] ** 2
    l2 = k_idx[None, :, None] ** 2
    m2 = k_idx[None, None, :] ** 2
    atten = np.exp(-D * (k2 + l2 + m2) * np.pi**2 * T)  # (K, K, K)

    u0s, uTs = [], []
    for _ in range(N):
        c = np.random.uniform(-20, 20, size=(K, K, K))  # coefficients

        u0 = np.einsum("klm,kx,ly,mz->xyz", c, sin_x, sin_y, sin_z)
        uT = np.einsum("klm,kx,ly,mz->xyz", c * atten, sin_x, sin_y, sin_z)

        u0s.append(u0.astype(np.float32))
        uTs.append(uT.astype(np.float32))

    return np.array(u0s), np.array(uTs)


class DiffusionDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {"x": self.X[idx].to(device), "y": self.Y[idx].to(device)}


# parametres
N, D, T, nx, K = 1000, 0.01, 0.1, 32, 8
print("Génération des données 3D...", flush=True)
u0s, uTs = dataset(N, D, T, nx, K)
print(f"Shape : {u0s.shape}", flush=True)  

X_t = torch.tensor(u0s).unsqueeze(1)  
Y_t = torch.tensor(uTs).unsqueeze(1)

train_dataset = DiffusionDataset(X_t[:800], Y_t[:800])
test_dataset = DiffusionDataset(X_t[800:], Y_t[800:])


train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loaders = {"test": DataLoader(test_dataset, batch_size=8, shuffle=False)}

model = FNO(
    n_modes=(N_MODES, N_MODES, N_MODES),
    in_channels=1,
    out_channels=1,
    hidden_channels=HIDDEN,
    n_layers=N_LAYERS,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
train_loss = LpLoss(d=3, p=2)  
eval_losses = {"L2": LpLoss(d=3, p=2)}

print(
    f"Config {CONFIG} : n_modes={N_MODES}, hidden={HIDDEN}, n_layers={N_LAYERS}",
    flush=True,
)

trainer = Trainer(
    model=model,
    n_epochs=N_EPOCHS,
    device=str(device),
    verbose=True,
)

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

file_exists = os.path.isfile(RESULTS_FILE)
with open(RESULTS_FILE, "a", newline="") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(
            [
                "config",
                "model_path",
                "n_modes",
                "hidden_channels",
                "n_layers",
                "n_epochs",
                "energy_train_joules",
            ]
        )
    writer.writerow(
        [CONFIG, MODEL_PATH, N_MODES, HIDDEN, N_LAYERS, N_EPOCHS, energy_train_joules]
    )
    fcntl.flock(f, fcntl.LOCK_UN)

print(f"Résultats enregistrés dans {RESULTS_FILE}", flush=True)
torch.save(model, MODEL_PATH)
print(f"Modèle sauvegardé : {MODEL_PATH}", flush=True)
