# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np

def convert2ras(data):
    """Convert dataset from original space into RAS space.
    Original space:
        axis i: A->P
        axis j: I->S
        axis k: R->L
    """
    # reverse axis i (A-P to P-A)
    ndata = data[::-1, :, :]
    # reverse axis k (R-L to L-R)
    #ndata = ndata[:, :, ::-1]
    # convert matrix into RAS space
    ndata = np.rollaxis(ndata, 2)
    return ndata

def idx2coord(vec_idx):
    """Convert row index in response data matrix into 3D coordinate in
    (original) ROI volume.
    """
    data_size = (18, 64, 64)
    coord_z = vec_idx % data_size[2]
    coord_x = vec_idx / (data_size[1]*data_size[2])
    coord_y = (vec_idx % (data_size[1]*data_size[2])) / data_size[2]
    return (coord_x, coord_y, coord_z)

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

def node2feature(layer_name, node_idx):
    """Convert node index from CNN activation vector into 3 features including
    index of channel, row and column position of the filter.
    Return a tuple of (channel index, row index, column index).
    """
    data_size = {'conv1': [96, 55, 55],
                 'conv2': [256, 27, 27],
                 'conv3': [384, 13, 13],
                 'conv4': [384, 13, 13],
                 'cpnv5': [256, 13, 13],
                 'pool5': [256, 6, 6]}
    s = data_size[layer_name]
    col_idx = node_idx % s[2]
    channel_idx = node_idx / (s[1]*s[2])
    row_idx = (node_idx % (s[1]*s[2])) / s[2]
    return (channel_idx, row_idx, col_idx)

