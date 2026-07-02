import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, plot
import ufl
from dolfinx.fem.petsc import LinearProblem
import pyvista

#  Maillage
domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)

# Espace fonctionnel
V = fem.functionspace(domain, ("Lagrange", 1))

# Paramètres physiques 
kappa = 0.01
dt = 0.01
T = 1.0

# Condition initiale : sin(πx)sin(πy)
u_n = fem.Function(V)
u_n.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

#  Pas de source
f = fem.Constant(domain, 0.0)

# Conditions aux limites de Dirichlet : u = 0 sur ∂Ω
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

# Préparation pyvista
tdim = domain.topology.dim
domain.topology.create_connectivity(tdim, tdim)
topology, cell_types, geometry = plot.vtk_mesh(V)
grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

# Historique des erreurs
historique_t = []
historique_L2 = []
historique_max = []

#  Boucle en temps
plotter = pyvista.Plotter()
plotter.show(interactive_update=True)

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
        lambda x: np.exp(-2 * np.pi**2 * kappa * t)
        * np.sin(np.pi * x[0])
        * np.sin(np.pi * x[1])
    )

    # Erreur L2
    L2_form = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
    error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(L2_form), op=MPI.SUM))

    # Erreur max
    error_max = np.max(np.abs(uh.x.array - uex.x.array))

    # Mise à jour u_n
    u_n.x.array[:] = uh.x.array

    historique_t.append(t)
    historique_L2.append(error_L2)
    historique_max.append(error_max)

    if round(t / dt) % 10 == 0:
        print(
            f"t={t:.2f} | u_max={uh.x.array.max():.4f} | erreur L2={error_L2:.2e} | erreur max={error_max:.2e}"
        )

        grid.point_data["u"] = uh.x.array
        grid.set_active_scalars("u")
        plotter.clear()
        plotter.add_mesh(grid.copy(), show_edges=False, clim=[0, 1.0])
        plotter.view_xy()
        plotter.update()
        time.sleep(0.2)

plotter.close()

# Résumé final 
print(f"\nErreur L2 moyenne  : {np.mean(historique_L2):.2e}")
print(f"Erreur max moyenne : {np.mean(historique_max):.2e}")
