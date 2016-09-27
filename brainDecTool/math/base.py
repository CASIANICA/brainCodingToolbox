# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np

def corr2_coef(A, B):
    """Row-wise Correlation Coefficient calculation for two 2D arrays."""
    # Row-wise mean of input arrays & subtract from input arrays themselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)
    # Finally get corr coef
    return np.dot(A_mA, B_mB.T)/np.sqrt(np.dot(ssA[:, None], ssB[None]))

