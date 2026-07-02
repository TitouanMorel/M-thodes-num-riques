import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.io import XDMFFile
import ufl
from dolfinx.fem.petsc import LinearProblem

# Maillage
domain = mesh.create_unit_cube(MPI.COMM_WORLD, 16, 16, 16)

# Espace fonctionnel
V = fem.functionspace(domain, ("Lagrange", 1))

# Paramètres physiques
kappa = 0.005
dt = 0.01
T = 5.0

# Condition initiale
u_n = fem.Function(V)
u_n.interpolate(
    lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])
)

# Pas de source
f = fem.Constant(domain, 0.0)

# Conditions aux limites
uD = fem.Function(V)
uD.interpolate(lambda x: np.zeros(x.shape[1]))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

# Fonctions test et trial
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Formulation variationnelle
a = (u / dt) * v * ufl.dx + kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n / dt) * v * ufl.dx + f * v * ufl.dx

# Historique des erreurs
historique_L2 = []
historique_max = []

# # Export XDMF : écriture du maillage une seule fois
# xdmf = XDMFFile(domain.comm, "solution.xdmf", "w")
# xdmf.write_mesh(domain)

# Boucle en temps
t = 0.0
while t < T:
    t += dt

    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="diffusion",
    )
    uh = problem.solve()

    # Solution exacte à l'instant t
    uex = fem.Function(V)
    uex.interpolate(
        lambda x: np.exp(-3 * np.pi**2 * kappa * t)
        * np.sin(np.pi * x[0])
        * np.sin(np.pi * x[1])
        * np.sin(np.pi * x[2])
    )

    # Erreur L2
    L2_form = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(L2_form), op=MPI.SUM))

    # Erreur max
    error_max = np.max(np.abs(uh.x.array - uex.x.array))

    # Mise à jour u_n
    u_n.x.array[:] = uh.x.array

    # Export de la solution à cet instant
    # xdmf.write_function(uh, t)

    historique_L2.append(error_L2)
    historique_max.append(error_max)
# xdmf.close()

# Résumé final
print(f"\nErreur L2 moyenne  : {np.mean(historique_L2):.2e}")
print(f"Erreur max moyenne : {np.mean(historique_max):.2e}")
# print("Fichier sauvegardé : solution.xdmf")
