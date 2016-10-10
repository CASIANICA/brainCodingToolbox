# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib

def idx2coord(vec_idx):
    """Convert row index in response data matrix into 3D coordinate in
    (original) ROI volume.
    """
    data_size = (18, 64, 64)
    coord_z = vec_idx % data_size[2]
    coord_x = vec_idx / (data_size[1]*data_size[2])
    coord_y = (vec_idx % (data_size[1]*data_size[2])) / data_size[2]
    return (coord_x, coord_y, coord_z)

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

def coord2ecc(pos_mtx):
    """Return the distance to the image center."""
    center_pos = [27, 27]
    row_num = pos_mtx.shape[0]
    cen = np.repeat([center_pos], row_num, axis=0)
    return np.linalg.norm(pos_mtx-cen, axis=1)

def coord2angle(pos_mtx):
    """Return the angle given a coordinate in the image."""
    std_pos = [0, 27]
    center_pos = [27, 27]
    std_vtr = np.array(std_pos) - np.array(center_pos)
    row_num = pos_mtx.shape[0]
    cen = np.repeat([center_pos], row_num, axis=0)
    vtr = pos_mtx - cen
    ang = np.zeros(row_num)
    for i in range(row_num):
        uvtr = unit_vector(vtr[i])
        usvtr = unit_vector(std_vtr)
        if uvtr[0] < 0:
            flag = -1
        else:
            flag = 1
        ang[i] = np.arccos(np.clip(np.dot(uvtr, usvtr), -1.0, 1.0)) * flag
    return ang

def unit_vector(vector):
    """Return the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def save2nifti(data, filename):
    """Save 3D data as nifti file.
    Original data shape is (18, 64, 64), and the resulting data shape is
    (64, 64, 18) which orientation is SLP."""
    # roll axis
    ndata = np.rollaxis(data, 0, 3)
    # generate affine matrix
    aff = np.zeros((4, 4))
    aff[0, 1] = -2
    aff[1, 2] = -2.5
    aff[2, 0] = 2
    aff[3, 3] = 1
    img = nib.Nifti1Image(ndata, aff)
    nib.save(img, filename)

