# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from skimage.measure import block_reduce

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

def unit_vector(vector):
    """Return the unit vector of the input."""
    return vector / np.linalg.norm(vector)

def down_sample(image, block_size, cval=0):
    """Down-sampling an input image, and the unit of down-sample is specified
    with `block_size`.
    `cval` : Constant padding value is image is not perfectly divisible by
    the block size.

    Example
    -------
    image shape : 10, 10, 8
    block_size : 2, 2, 1
    the down_sample(image, block_size=(2, 2, 1)) would return an image which 
    shape is (5, 5, 8).
    """
    return block_reduce(image, block_size, func=np.mean, cval=cval)

