import numpy as np
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker
from ecartmoyen import (
    solve_fenicsx_implicite,
    solve_fenicsx_implicite_3D,
    create_L_mesh,
    compute_reference_Lshape,
    solve_fenicsx_implicite_Lshape,
)

# Paramètres fixes
kappa = 0.01
T_final = 5.0
dt_valeurs = T_final / np.array([5, 10, 20, 40, 80, 160, 320, 640, 1280])


def mesurer_energie(fn, *args):
    tracker = EmissionsTracker(
        save_to_file=False,
        logging_logger=None,
        measure_power_secs=0.1,
    )
    tracker.start()
    resultat = fn(*args)
    tracker.stop()
    energie_J = tracker.final_emissions_data.energy_consumed * 3.6e6
    return resultat, energie_J


#  2D
nx, ny = 64, 64
ecarts_2D = np.zeros(len(dt_valeurs))
energie_2D = np.zeros(len(dt_valeurs))

for i, dt in enumerate(dt_valeurs):
    print(f"[2D] dt={dt:.5f} ...", end=" ", flush=True)
    ecarts_2D[i], energie_2D[i] = mesurer_energie(
        solve_fenicsx_implicite, nx, ny, kappa, T_final, dt
    )
    print(f"erreur={ecarts_2D[i]:.2e}  énergie={energie_2D[i]:.3f} J")

# 3D
nx, ny, nz = 16, 16, 16
ecarts_3D = np.zeros(len(dt_valeurs))
energie_3D = np.zeros(len(dt_valeurs))

for i, dt in enumerate(dt_valeurs):
    print(f"[3D] dt={dt:.5f} ...", end=" ", flush=True)
    ecarts_3D[i], energie_3D[i] = mesurer_energie(
        solve_fenicsx_implicite_3D, nx, ny, nz, kappa, T_final, dt
    )
    print(f"erreur={ecarts_3D[i]:.2e}  énergie={energie_3D[i]:.3f} J")

# L
print("\n[L-shape] création du maillage...", flush=True)
domain_L, V_L = create_L_mesh(h=0.01)

print("[L-shape] calcul de la référence...", flush=True)
u_ref_L = compute_reference_Lshape(
    kappa, T_final, domain_L, V_L, n_sub=100, dt_base=dt_valeurs[-1]
)

ecarts_L = np.zeros(len(dt_valeurs))
energie_L = np.zeros(len(dt_valeurs))

for i, dt in enumerate(dt_valeurs):
    print(f"[L-shape] dt={dt:.5f} ...", end=" ", flush=True)
    ecarts_L[i], energie_L[i] = mesurer_energie(
        solve_fenicsx_implicite_Lshape, kappa, T_final, dt, domain_L, V_L, u_ref_L
    )
    print(f"erreur={ecarts_L[i]:.2e}  énergie={energie_L[i]:.3f} J")

eps_values = np.logspace(-1, -6, 200)
cout_2D = np.full(len(eps_values), np.nan)
cout_3D = np.full(len(eps_values), np.nan)
cout_L = np.full(len(eps_values), np.nan)

for j, eps in enumerate(eps_values):
    candidats_2D = [energie_2D[i] for i in range(len(dt_valeurs)) if ecarts_2D[i] < eps]
    candidats_3D = [energie_3D[i] for i in range(len(dt_valeurs)) if ecarts_3D[i] < eps]
    candidats_L = [energie_L[i] for i in range(len(dt_valeurs)) if ecarts_L[i] < eps]
    if candidats_2D:
        cout_2D[j] = min(candidats_2D)
    if candidats_3D:
        cout_3D[j] = min(candidats_3D)
    if candidats_L:
        cout_L[j] = min(candidats_L)

plt.figure()
plt.plot(eps_values, cout_2D, label=f"FEM 2D ({nx}×{ny})")
plt.plot(eps_values, cout_3D, label=f"FEM 3D ({nx}×{ny}×{nz})")
plt.plot(eps_values, cout_L, label="FEM L-shape")
plt.xlabel("Précision cible ε")
plt.ylabel("Énergie consommée (J)")
plt.xscale("log")
plt.gca().invert_xaxis()
plt.title("FEniCSx — énergie vs précision (CodeCarbon)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
