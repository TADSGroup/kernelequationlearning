from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import FunctionSpace
from dolfinx import fem
import ufl
import numpy as np
from dataclasses import dataclass

from petsc4py.PETSc import ScalarType
from scipy.interpolate import RectBivariateSpline
from dolfinx import geometry

def zero_boundary_conditions(x):
    """interpolated zero boundary conditions

    Parameters
    ----------
    x : array
        variable on domain

    Returns
    -------
    scalar output
        zero
    """
    return 0*x[0]


@dataclass
class DarcySolver():
    """
    Wrapper for solving darcy equation
    """
    num_x_cells: int = 100
    num_y_cells: int = 100
    function_space_type: str = "CG"
    function_space_degree: int = 2

    bc_func= zero_boundary_conditions

    solution = None

    def __post_init__(self):
        self.domain = mesh.create_unit_square(
            MPI.COMM_WORLD,
            self.num_x_cells,
            self.num_y_cells
        )

        self.V = FunctionSpace(
            self.domain,
            (self.function_space_type, self.function_space_degree)
        )

        bcDirichlet = fem.Function(self.V)
        bcDirichlet.interpolate(zero_boundary_conditions)

        # Create facet to cell connectivity required to determine boundary facets
        tdim = self.domain.topology.dim
        fdim = tdim - 1
        self.domain.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(self.domain.topology)
        boundary_dofs = fem.locate_dofs_topological(self.V, fdim, boundary_facets)
        self.bc = fem.dirichletbc(bcDirichlet, boundary_dofs)

    def solve(
        self,
        permeability,
        f_rhs,
        num_interpolation_points=100,
        ks_type= 'preonly',
        pc_type= 'lu'
    ):
        """
        solves darcy flow
        Interpolates the functions permeability and f_rhs
            to input to solver

        Parameters
        ----------
        permeability : callable
            permeability
        f_rhs : callable
            source term
        interpolation_points: int
            number of points to interpolate the final solution at in each direction
        """
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        f = fem.Function(self.V)
        f.interpolate(f_rhs)

        perm_coef=fem.Function(self.V)
        perm_coef.interpolate(permeability)

        A = perm_coef*ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        rhs = f * v * ufl.dx

        problem = fem.petsc.LinearProblem(A, rhs, bcs=[self.bc], petsc_options={
                                          "ksp_type": ks_type, "pc_type": pc_type})
        solution = problem.solve()
        self.solution = solution

        return self.interpolate_function(solution, num_interpolation_points)

    def _get_cells(self, points):
        if points.shape[1] == 2:
            eval_points = np.hstack([points, np.zeros((len(points), 1))])
        elif points.shape[1] == 3:
            eval_points = points.copy()

        bb_tree = geometry.BoundingBoxTree(
            self.domain, self.domain.topology.dim)
        cells = []
        # Find cells whose bounding-box collide with the the points
        cell_candidates = geometry.compute_collisions(bb_tree, eval_points)

        # Compute the cells that contain each point
        colliding_cells = geometry.compute_colliding_cells(
            self.domain,
            cell_candidates,
            eval_points
        )
        # Choose one of the cells that contains the point
        cells = [colliding_cells.links(i)[0] for i in range(len(eval_points))]
        return eval_points, cells

    def interpolate_function(self, u, num_interpolation_points=200):
        """Interpolates the fem.Function u to give a callable representation

        Parameters
        ----------
        u : fem.Function
        num_interpolation_points : int, optional
            number of points to interpolate with in each direction, by default 200
        """
        xgrid = np.linspace(0, 1, num_interpolation_points)
        ygrid = np.linspace(0, 1, num_interpolation_points)
        X, Y = np.meshgrid(xgrid, ygrid)
        evaluation_points = np.array(
            [[x, y] for x, y in zip(X.flatten(), Y.flatten())])
        eval_points, cells = self._get_cells(evaluation_points)
        function_values = u.eval(eval_points, cells).reshape(
            num_interpolation_points, num_interpolation_points)
        interpolated_function = RectBivariateSpline(
            xgrid, ygrid, function_values)
        return interpolated_function
