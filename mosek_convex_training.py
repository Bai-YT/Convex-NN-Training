# Unofficial implementation of convex neural network training
# Yatong Bai
# October 24th, 2022
# Requires numpy, cvxpy, mosek


import numpy as np
from numpy.random import randn
import cvxpy as cp
import mosek


def relu_prime(z):
    """
    Returns the derivative of the ReLU activation function
    """
    return (z >= 0).astype(int)


def generate_D(X, P, v=-1, w=-1, verbose=False):
    """
    Generates the D_i matrices required for convex training.

    :param X:       Training data with dimension (n, d).
    :param P:       Number of sampled hyperplanes.
    :param v:       The weights being used to generate the D_i matrices (hyperplane arrangements).
                    If -1, then use random weights. Default: -1.
    :param w:       The weights being used to generate the D_i matrices (hyperplane arrangements).
                    If -1, then use random weights. Default: -1.
    :param verbose: If true, intermediate check results will be printed.

    :return:        dmat, n, d, P, v, w
    """
    (n, d) = X.shape
    X = X.astype(np.float32)
    if w == -1 and v == -1:
        v = randn(d, P).astype(np.float32)
        dmat, ind = np.unique(relu_prime(X @ v), axis=1, return_index=True)
        v = v[:, ind]
        if verbose:
            print((2 * dmat-1) * (X @ v) >= 0)
    else:
        P = v.shape[1]
        dmat1 = relu_prime(X @ v)
        dmat2 = relu_prime(X @ w)
        dmat = np.concatenate([dmat1, dmat2], axis=1)
        temp, ind = np.unique(dmat, axis=1, return_index=True)
        ind1 = ind[ind < P]
        ind2 = ind[ind >= P] - P
        dmat = dmat[:, np.concatenate([ind1, ind2+P])]
        wnew = w[:, ind2]
        v, w = v[:, ind1], w[:, ind1]
        w[:, ind2] = np.zeros([d, ind2.size])
        w = np.concatenate([w, wnew], axis=1)
        v = np.concatenate([v, np.zeros([d, ind2.size])], axis=1)
        if verbose:
            print((2 * dmat - 1) * (X @ v) >= 0)
            print((2 * dmat - 1) * (X @ w) >= 0)
    return dmat, n, d, dmat.shape[1], v, w


def nnfit_cvx(X, y, P, beta=1e-4, dmat=-1, solver=cp.MOSEK, loss_type='mse', verbose=True):
    """
    Performs convex training of one-hidden-layer neural networks.

    :param X:           Training data with dimension (n, d).
    :param y:           Training targets with dimension (n,).
    :param P:           Number of sampled hyperplanes.
    :param beta:        The regularization strength. Default is 1e-4.
    :param dmat:        The D_i matrices. If -1, then randomly generate. Default is -1.
    :param solver:      Specifies the CVX solver. Default is MOSEK.
    :param loss_type:   The loss function type. Must be either 'mse' or 'bce'. Default is 'mse'
                        Note that convex training applies to all convex loss functions.
    :param verbose:     If true, the status of the solver will be displayed.

    :return:            v_star, w_star, optimal_objective, d_matrices,
                        where v_star and w_star are the optimal weights.
    """

    print('Generating D matrices...')
    if dmat == -1:
        dmat, n, d, P, v, w = generate_D(X, P)
    else:
        (n, d), (_, P) = X.shape, dmat.shape
    emat = 2 * dmat - np.ones((n, P))

    # Optimal CVX
    print('Building CVX problem...')
    uopt1 = cp.Variable((d, P))
    uopt2 = cp.Variable((d, P))

    yhat = cp.sum(cp.multiply(dmat, (X @ (uopt1 - uopt2))), axis=1)
    if loss_type == 'mse':  # Squared loss
        cost = cp.sum_squares(yhat - y) / 2 + \
               beta * cp.mixed_norm(uopt1.T, 2, 1) + beta * cp.mixed_norm(uopt2.T, 2, 1)
    elif loss_type == 'bce':  # Binary cross entropy
        cost = cp.sum(cp.logistic(2 * yhat) - 2 * cp.multiply(y, yhat)) + \
               beta * cp.mixed_norm(uopt1.T, 2, 1) + beta * cp.mixed_norm(uopt2.T, 2, 1)
    else:
        raise ValueError("Unknown loss function.")
    constraints = [cp.multiply(emat, (X @ uopt1)) >= 0]
    constraints += [cp.multiply(emat, (X @ uopt2)) >= 0]

    print('Solving CVX problem...')
    prob = cp.Problem(cp.Minimize(cost), constraints)
    if solver is cp.MOSEK:
        prob.solve(solver=solver, verbose=verbose, mosek_params={mosek.dparam.intpnt_co_tol_mu_red: 5e-14,
                                                                 mosek.dparam.intpnt_co_tol_rel_gap: 5e-14,
                                                                 mosek.dparam.intpnt_co_tol_infeas: 1e-14})
    else:
        prob.solve(solver=solver, verbose=verbose)

    print("\nTotal cost: ", prob.value)
    return uopt1.value, uopt2.value, prob.value, dmat
