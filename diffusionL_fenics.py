import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
from dolfinx import mesh, fem, io
import ufl
from dolfinx.fem.petsc import LinearProblem
import gmsh
import meshio



def create_L_mesh(h=0.03):
    gmsh.initialize()
    gmsh.model.add("L-shape")
    r1 = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1)
    r2 = gmsh.model.occ.addRectangle(0.5, 0.5, 0, 0.5, 0.5)
    gmsh.model.occ.cut([(2, r1)], [(2, r2)])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(2)
    gmsh.write("L_shape.msh")
    gmsh.finalize()

    msh = meshio.read("L_shape.msh")
    cells = msh.cells_dict["triangle"]
    meshio.write(
        "L_shape_mesh.xdmf",
        meshio.Mesh(points=msh.points[:, :2], cells=[("triangle", cells)]),
    )
    with io.XDMFFile(MPI.COMM_WORLD, "L_shape_mesh.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    return domain


if __name__ == "__main__":
    domain = create_L_mesh(h=0.03)
    domain.topology.create_connectivity(domain.topology.dim, domain.topology.dim - 1)
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)

    V = fem.functionspace(domain, ("Lagrange", 1))

    dof_coords = V.tabulate_dof_coordinates()[:, :2]
    n_cells    = domain.topology.index_map(domain.topology.dim).size_local
    cells_dofs = np.array([V.dofmap.cell_dofs(i) for i in range(n_cells)])
    triang     = mtri.Triangulation(dof_coords[:, 0], dof_coords[:, 1], cells_dofs)

    kappa  = 0.01
    dt     = 0.005
    n_sub  = 20
    dt_ref = dt / n_sub
    T      = 1.0

    def u0(x):
        return 300.0 + 100.0 * np.exp(-((x[0] - 0.3)**2 + (x[1] - 0.3)**2) / 0.08)

    u_n = fem.Function(V)
    u_n.interpolate(u0)

    u_n_ref = fem.Function(V)
    u_n_ref.interpolate(u0)

    f = fem.Constant(domain, 0.0)
    uD = fem.Function(V)
    uD.interpolate(lambda x: 300.0 * np.ones(x.shape[1]))
    fdim            = domain.topology.dim - 1
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs   = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a     = (u/dt)     * v * ufl.dx + kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L     = (u_n/dt)   * v * ufl.dx + f * v * ufl.dx
    a_ref = (u/dt_ref) * v * ufl.dx + kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_ref = (u_n_ref/dt_ref) * v * ufl.dx + f * v * ufl.dx

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tpc = axes[0].tripcolor(
        triang, u_n.x.array, cmap="hot", shading="gouraud", vmin=300.0, vmax=400.0
    )
    plt.colorbar(tpc, ax=axes[0])
    axes[0].set_aspect("equal")
    axes[0].set_title("u(x,y,t=0.00)")

    (line,) = axes[1].semilogy([], [], "b-")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("erreur L2 relative")
    axes[1].set_title("Erreur vs solution de référence")
    axes[1].set_xlim(0, T)
    axes[1].set_ylim(1e-6, 1e0)
    axes[1].grid(True, which="both")

    plt.tight_layout()
    plt.show()

    historique_t  = []
    historique_L2 = []

    t = 0.0
    while t < T:
        t += dt

        uh = LinearProblem(a, L, bcs=[bc],
                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                           petsc_options_prefix="coarse").solve()
        u_n.x.array[:] = uh.x.array

        for _ in range(n_sub):
            uh_ref = LinearProblem(a_ref, L_ref, bcs=[bc],
                                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                   petsc_options_prefix="ref").solve()
            u_n_ref.x.array[:] = uh_ref.x.array

        err = fem.Function(V)
        err.x.array[:] = uh.x.array - uh_ref.x.array

        num = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(err, err) * ufl.dx)), op=MPI.SUM))
        den = np.sqrt(domain.comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(uh_ref, uh_ref) * ufl.dx)), op=MPI.SUM))

        error_L2 = num / den if den > 0.0 else 0.0
        historique_t.append(t)
        historique_L2.append(error_L2)

        if round(t / dt) % 5 == 0:
            print(f"t={t:.2f} | u_max={uh.x.array.max():.4f} | erreur L2 rel={error_L2:.2e}", flush=True)
            tpc.set_array(uh.x.array)
            axes[0].set_title(f"u(x,y,t={t:.2f})")
            line.set_data(historique_t, historique_L2)
            axes[1].set_xlim(0, T)
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)

    print(f"\nErreur L2 relative finale vs référence : {historique_L2[-1]:.2e}")
    print(f"Erreur L2 relative max sur [0, T]      : {max(historique_L2):.2e}")

    plt.ioff()
    plt.savefig("solution_finale.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : solution_finale.png")