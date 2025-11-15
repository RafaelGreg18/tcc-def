import numpy as np
from scipy.stats import wasserstein_distance


def wasserstein_distances(A):
    n, m = A.shape
    uniform = np.full(m, 1 / m)
    classes = np.arange(m)

    dists = []
    for i in range(n):
        row = A[i]
        if row.sum() == 0:
            continue
        p = row / row.sum()
        # scipy aceita pesos (distribuições) sobre posições discretas
        d = wasserstein_distance(classes, classes, p, uniform)
        dists.append(d)

    avg_dist = np.mean(dists)

    # Distribuição global
    totals = A.sum(axis=0)
    P = totals / totals.sum()
    global_dist = wasserstein_distance(classes, classes, P, uniform)

    return avg_dist, global_dist, dists, P


# Exemplo
A = np.array([
    [10, 0, 0],
    [0, 5, 5],
    [2, 2, 6]
])

avg_dist, global_dist, dists, P = wasserstein_distances(A)
print("Distâncias individuais:", dists)
print("Distância média clientes:", avg_dist)
print("Distribuição global:", P)
print("Distância global p/ uniforme:", global_dist)
