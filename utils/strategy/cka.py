import numpy as np


def cka_wide(X, Y):
    """
    Calculate CKA for two matrices. This algorithm uses a Gram matrix
    implementation, which is fast when the data is wider than it is
    tall.

    This implementation is inspired by the one in this colab:
    https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ

    Note that we use center the features rather than the Gram matrix
    because we think the latter is tricky and mysterious. It only works for
    linear CKA though (we only implement linear CKA throughout).
    """
    X = X.copy()
    Y = Y.copy()

    X -= X.mean(0)
    Y -= Y.mean(0)

    XXT = X @ X.T
    YYT = Y @ Y.T

    # We use reshape((-1,)) instead of ravel() to ensure this is compatible
    # with numpy and pytorch tensors.
    top = (XXT.reshape((-1,)) * YYT.reshape((-1,))).sum()
    bottom = np.sqrt((XXT ** 2).sum() * (YYT ** 2).sum())
    c = top / bottom

    return c


def cka_tall(X, Y):
    """
    Calculate CKA for two matrices.
    """
    X = X.copy()
    Y = Y.copy()

    X -= X.mean(0)
    Y -= Y.mean(0)

    XTX = X.T @ X
    YTY = Y.T @ Y
    YTX = Y.T @ X

    # Equation (4)
    top = (YTX ** 2).sum()
    bottom = np.sqrt((XTX ** 2).sum() * (YTY ** 2).sum())
    c = top / bottom

    return c


def cka(X, Y):
    """
    Calculate CKA for two matrices.

    CKA has several potential implementations. The naive implementation is
    appropriate for tall matrices (more examples than features), but this
    implementation uses lots of memory and it slow when there are many more
    features than examples. In that case, which often happens with DNNs, we
    prefer the Gram matrix variant.
    """
    if X.shape[0] < X.shape[1]:
        return cka_wide(X, Y)
    else:
        return cka_tall(X, Y)


def compute_model_heterogeneity(client_models):
    if len(client_models) < 2:
        return 0.0

    mats = []
    for params in client_models:
        flat = np.concatenate([arr.flatten() for arr in params])
        mats.append(flat.reshape(1, -1))

    n = len(mats)
    sims = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                X, Y = mats[i].T, mats[j].T
                sim = cka(X, Y)
                if np.isnan(sim) or np.isinf(sim):
                    corr = np.corrcoef(mats[i].flatten(), mats[j].flatten())[0, 1]
                    sim = float(abs(corr)) if not np.isnan(corr) else 0.0
                else:
                    sim = float(sim)
                sims.append(sim)
            except Exception:
                try:
                    corr = np.corrcoef(mats[i].flatten(), mats[j].flatten())[0, 1]
                    sim = float(abs(corr)) if not np.isnan(corr) else 0.0
                    sims.append(sim)
                except Exception:
                    sims.append(0.0)

    if not sims:
        return 0.0

    avg_sim = float(np.mean(sims))
    het = 1.0 - avg_sim
    return float(np.clip(het, 0.0, 1.0))
