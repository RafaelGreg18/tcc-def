from itertools import product

import numpy as np


def get_gradients(global_m, local_models, num_clients):
    """return the `representative gradient` formed by the difference between
    the local work and the sent global model"""

    local_model_params = []

    if len(local_models) == 0:
        for cid in range(num_clients):
            local_model_params += [
                [tens.detach().cpu().numpy() for tens in list(global_m.parameters())]
            ]
    else:
        for model in local_models:
            local_model_params += [
                [tens.detach().cpu().numpy() for tens in list(model.parameters())]
            ]

    global_model_params = [
        tens.detach().cpu().numpy() for tens in list(global_m.parameters())
    ]

    local_model_grads = []
    for local_params in local_model_params:
        local_model_grads += [
            [
                local_weights - global_weights
                for local_weights, global_weights in zip(
                local_params, global_model_params
            )
            ]
        ]

    return local_model_grads


def submod_sampling(gradients, n_sampled, n_available, stochastic=False):
    norm_diff = compute_diff(gradients, "euclidean")
    np.fill_diagonal(norm_diff, 0)
    indices = select_cl_submod(num_clients=n_sampled, num_available=n_available, norm_diff=norm_diff, stochastic=stochastic)
    return indices


def compute_diff(gradients, metric):
    n_clients = len(gradients)

    metric_matrix = np.zeros((n_clients, n_clients))
    for i, j in product(range(n_clients), range(n_clients)):
        metric_matrix[i, j] = get_similarity(
            gradients[i], gradients[j], metric
        )

    return metric_matrix


def select_cl_submod(num_clients, num_available, norm_diff, stochastic=False):
    if stochastic:
        SUi = stochastic_greedy(norm_diff, num_clients, num_available)
    else:
        SUi = lazy_greedy(norm_diff, num_clients, num_available)
    indices = np.array(list(SUi))
    return indices

def get_similarity(grad_1, grad_2, distance_type="L1"):
    if distance_type == "L1":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum(np.abs(g_1 - g_2))
        return norm

    elif distance_type == "euclidean":
        norm = 0
        for g_1, g_2 in zip(grad_1, grad_2):
            norm += np.sum((g_1 - g_2) ** 2)
        return np.sqrt(norm)

    elif distance_type == "cosine":
        norm, norm_1, norm_2 = 0, 0, 0
        for i in range(len(grad_1)):
            norm += np.sum(grad_1[i] * grad_2[i])
            norm_1 += np.sum(grad_1[i] ** 2)
            norm_2 += np.sum(grad_2[i] ** 2)

        if norm_1 == 0.0 or norm_2 == 0.0:
            return 0.0
        else:
            norm /= np.sqrt(norm_1 * norm_2)

            return np.arccos(norm)

def stochastic_greedy(norm_diff, num_clients, num_available):
    # initialize the ground set and the selected set
    V_set = set(range(num_available))
    SUi = set()

    m = num_clients
    for ni in range(num_clients):
        if m < len(V_set):
            R_set = np.random.choice(list(V_set), m, replace=False)
        else:
            R_set = list(V_set)
        if ni == 0:
            marg_util = norm_diff[:, R_set].sum(0)
            i = marg_util.argmin()
            client_min = norm_diff[:, R_set[i]]
        else:
            client_min_R = np.minimum(client_min[:, None], norm_diff[:, R_set])
            marg_util = client_min_R.sum(0)
            i = marg_util.argmin()
            client_min = client_min_R[:, i]
        SUi.add(R_set[i])
        V_set.remove(R_set[i])
    return SUi


def lazy_greedy(norm_diff, num_clients, num_available):
    # initialize the ground set and the selected set
    V_set = set(range(num_available))
    SUi = set()

    S_util = 0
    marg_util = norm_diff.sum(0)
    i = marg_util.argmin()
    L_s0 = 2. * marg_util.max()
    marg_util = L_s0 - marg_util
    client_min = norm_diff[:, i]
    # print(i)
    SUi.add(i)
    V_set.remove(i)
    S_util = marg_util[i]
    marg_util[i] = -1.

    while len(SUi) < num_clients:
        argsort_V = np.argsort(marg_util)[len(SUi):]
        for ni in range(len(argsort_V)):
            i = argsort_V[-ni - 1]
            SUi.add(i)
            client_min_i = np.minimum(client_min, norm_diff[:, i])
            SUi_util = L_s0 - client_min_i.sum()

            marg_util[i] = SUi_util - S_util
            if ni > 0:
                if marg_util[i] < marg_util[pre_i]:
                    if ni == len(argsort_V) - 1 or marg_util[pre_i] >= marg_util[argsort_V[-ni - 2]]:
                        S_util += marg_util[pre_i]
                        # print(pre_i, L_s0 - S_util)
                        SUi.remove(i)
                        SUi.add(pre_i)
                        V_set.remove(pre_i)
                        marg_util[pre_i] = -1.
                        client_min = client_min_pre_i.copy()
                        break
                    else:
                        SUi.remove(i)
                else:
                    if ni == len(argsort_V) - 1 or marg_util[i] >= marg_util[argsort_V[-ni - 2]]:
                        S_util = SUi_util
                        # print(i, L_s0 - S_util)
                        V_set.remove(i)
                        marg_util[i] = -1.
                        client_min = client_min_i.copy()
                        break
                    else:
                        pre_i = i
                        SUi.remove(i)
                        client_min_pre_i = client_min_i.copy()
            else:
                if marg_util[i] >= marg_util[argsort_V[-ni - 2]]:
                    S_util = SUi_util
                    # print(i, L_s0 - S_util)
                    V_set.remove(i)
                    marg_util[i] = -1.
                    client_min = client_min_i.copy()
                    break
                else:
                    pre_i = i
                    SUi.remove(i)
                    client_min_pre_i = client_min_i.copy()
    return SUi
