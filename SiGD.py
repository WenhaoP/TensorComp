import numpy as np
import scipy

# @profile
def build_reduced_problem(Pts, hessian_diag, weights, grad_dot_Pts, reduced_hessian_old, mu_reduced_old, L_reduced_old, sparse):
    """
    Reduce the minimization over the convex hull of the
    active set to the minimization over the unit probability simplex.
    """
    if (reduced_hessian_old is None) & (L_reduced_old is None) & (mu_reduced_old is None): # no need to recompute if Pts was not change
        if sparse:
            Pts = Pts.toarray()
        reduced_hessian = Pts.T * hessian_diag @ Pts
        mu_reduced, L_reduced = scipy.linalg.eigvalsh(reduced_hessian)[[0,-1]]
    else:
        reduced_hessian = reduced_hessian_old
        mu_reduced = mu_reduced_old
        L_reduced = L_reduced_old
    reduced_linear = grad_dot_Pts[:,None] - reduced_hessian @ weights
    return reduced_hessian, reduced_linear, mu_reduced, L_reduced

# @profile
def projection_simplex_sort(x, s=1.0):
    """
    Perform a projection onto the probability simplex of radius `s`
    using a sorting algorithm.
    """
    x = x.flatten()
    n = len(x)
    if (x.sum() == 1) & (x >= 0.0).all():
        return x
    v = x - x.max()
    u = np.sort(v)[::-1] 
    cssv = u.cumsum()
    rho = np.sum(u * np.arange(n) > (cssv - s))
    theta = (cssv[rho-1] - s) / (rho)
    w = np.clip(v - theta, 0.0, np.inf)
    w = w[:,None]

    return w

# @profile
def accelerated_simplex_gradient_descent_over_probability_simplex(x, y, gamma, L, M, B):
    """
    One step of minimizing an objective function over the unit probability simplex
    using Nesterov'saccelerated gradient descent.
    """
    x_old = x
    gradient_y = B + M @ y
    x = projection_simplex_sort(y - gradient_y / L)
    y = x + gamma * (x - x_old)

    return x, y