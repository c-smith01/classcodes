# Project 1&2 Code to Generate Additional Data for Submission to Project
import numpy as np
import cvxpy as cp
from itertools import combinations, product
import pickle

def generate_systematic_generator_matrix(k, n, low=-100, high=100):
    assert k < n, "k must be less than n"
    identity = np.eye(k)
    P = np.random.uniform(low, high, size=(k, n - k))

    # Ensure no column of G is all-zero
    while np.any(np.all(P == 0, axis=0)):
        P = np.random.uniform(low, high, size=(k, n - k))

    G = np.hstack((identity, P))
    return G, P

def solve_lp_for_tuple(G, a, b, X, epsilon, m):
    k, n = G.shape
    u = cp.Variable(k)
    
    # Objective function: maximize ε[0] * G[:, a]^T · u
    objective = cp.Maximize(epsilon[0] * G[:, a] @ u)
    constraints = []

    # Ranking indices from permutation ω
    x_indices = [a] + sorted(X) + [b]
    X_sorted = sorted(X)
    
    # Build constraints as per the paper
    for j, s in zip(X_sorted, epsilon[1:m]):
        constraint = (epsilon[0] * (G[:, a] @ u) - s * (G[:, j] @ u) <= 0)
        constraints.append(constraint)
    
    constraints.append(G[:, b] @ u == 1)

    # Additional constraints for other indices Y
    Y = [i for i in range(n) if i not in [a, b] + X_sorted]
    for j in Y:
        constraints += [
            G[:, j] @ u <= 1,
            -G[:, j] @ u <= 1
        ]

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve()
        return prob.value if prob.status == 'optimal' else float('-inf')
    except:
        return float('-inf')

def generate_dataset(n, k, m, num_samples):
    data = []
    labels = []
    for _ in range(num_samples):
        G, P = generate_systematic_generator_matrix(k, n)
        hm = compute_m_height_lp(G, m)
        if np.isfinite(hm) and hm >= 1:
            data.append(P)
            labels.append(hm)
    return np.array(data), np.array(labels)

data, labels = generate_dataset(n=9, k=5, m=3, num_samples=100)
with open('dataset_k5_m3.pkl', 'wb') as f:
    pickle.dump((data, labels), f)




