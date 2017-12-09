# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

def get_pls_components(x_scores, x_loadings):
    """Get Canonical Components from original data `X`, based on estimated
    scores `x_scores` and loadings `x_loadings`.
    """
    n = x_scores.shape[0]
    p = x_loadings.shape[0]
    n_components = x_scores.shape[1]
    ccs = np.zeros((n, p, n_components))
    # get CCs
    for i in range(n_components):
        score = x_scores[:, i]
        score = score.reshape(-1, 1)
        loading = x_loadings[:, i]
        loading = loading.reshape(-1, 1)
        xk = np.dot(score, loading.T)
        ccs[..., i] = xk
    return xk

def pls_regression_predict(pls2, X):
    """Get predictions using a trained PLS regression model."""
    nX = (X - pls2.x_mean_) / pls2.x_std_
    Y = np.zeros_like(pls2.predict(X))
    for i in range(pls2.n_components):
        x_scores = np.dot(nX, pls2.x_weights_[:, i])
        nX -= np.dot(np.expand_dims(x_scores, 1),
                     np.expand_dims(pls2.x_loadings_[:, i], 1).T)
        Y += np.dot(np.expand_dims(x_scores, 1),
                    np.expand_dims(pls2.y_loadings_[:, i], 1).T)
    Y = Y * pls2.y_std_ + pls2.y_mean_
    return Y

