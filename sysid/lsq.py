import optimistix as optx
from sysid.utils import Params, State, Action, Output, Residual, ResidualArgs


def least_squares(residual: Residual, solver: optx.AbstractLeastSquaresSolver, opt_params: Params, args: ResidualArgs, max_steps: int = 100, throw: bool = False):
    """Least squares function for system identification.

    :param residual: Residual function.
    :param solver: An instance of an optimistix least squares solver.
    :param opt_params: Optimizable parameters.
    :param args: Non-optimizable arguments passed to the residual function.
    :param max_steps: Maximum number of optimization steps.
    :param throw: Whether to throw an exception if the solver does not converge.
    :return: Solution of the least squares problem.
    """
    sol = optx.least_squares(residual, solver, opt_params, args=args, max_steps=max_steps, throw=throw)
    # new_sol = optx.Solution(sol.value, None, sol.aux, sol.stats, None)
    return sol