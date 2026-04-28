import cvxpy as cp


class SolverManager:
    """
    Generic CVXPY solver handler.
    Supports multiple solvers and fallback logic.
    """

    def __init__(self, solver_order=None, verbose=False):
        self.solver_order = solver_order or ["MOSEK", "ECOS", "SCS", "OSQP"]
        self.verbose = verbose

    def _solve_with(self, problem: cp.Problem, solver_name: str):
        """
        Solve using a specific solver.
        Returns success flag.
        """

        if solver_name == "MOSEK":
            problem.solve(
                solver=cp.MOSEK,
                verbose=self.verbose,
                mosek_params={
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-7,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-7,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
                }
            )

        elif solver_name == "SCS":
            problem.solve(
                solver=cp.SCS,
                verbose=self.verbose,
                eps=1e-4,
                max_iters=10000
            )

        elif solver_name == "ECOS":
            problem.solve(
                solver=cp.ECOS,
                verbose=self.verbose
            )

        elif solver_name == "OSQP":
            problem.solve(
                solver=cp.OSQP,
                verbose=self.verbose
            )

        else:
            raise ValueError(f"Unknown solver: {solver_name}")

        return problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

    def solve(self, problem: cp.Problem):
        """
        Try multiple solvers until success.

        Returns:
            dict with solver info
        """
        for solver_name in self.solver_order:
            try:
                if self.verbose:
                    print(f"\nTrying solver: {solver_name}")

                success = self._solve_with(problem, solver_name)

                if success:
                    return {
                        "success": True,
                        "solver": solver_name,
                        "problem": problem,
                        "status": problem.status,
                        "value": problem.value,
                    }

            except Exception as e:
                print(f"{solver_name} failed: {e}")

        sol = {
            "success": False,
            "solver": "FAILED",
            "problem": problem,
            "status": problem.status,
            "value": None,
        }

        return sol