# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import nibabel as nib
import matplotlib.pylab as plt

from brainDecTool.math import corr2_coef

def idx2coord(vec_idx):
    """Convert row index in response data matrix into 3D coordinate in
    (original) ROI volume.
    """
    data_size = (18, 64, 64)
    coord_z = vec_idx % data_size[2]
    coord_x = vec_idx / (data_size[1]*data_size[2])
    coord_y = (vec_idx % (data_size[1]*data_size[2])) / data_size[2]
    return (coord_x, coord_y, coord_z)

def coord2idx(coord):
    """Convert a 3D coordinate from nifti file into row index in response
    data matrix.
    Input must be a tuple.
    """
    ncoord = (coord[2], coord[0], 63-coord[1])
    return ncoord[2]+ncoord[0]*64*64+ncoord[1]*64

def node2feature(node_idx, data_shape):
    """Convert node index from CNN activation vector into 3 features including
    index of channel, row and column position of the filter.
    Return a tuple of (channel index, row index, column index).
    """
    #data_size = {'conv1': [96, 55, 55],
    #             'conv2': [256, 27, 27],
    #             'conv3': [384, 13, 13],
    #             'conv4': [384, 13, 13],
    #             'cpnv5': [256, 13, 13],
    #             'pool5': [256, 6, 6]}
    #s = data_size[layer_name]
    s = data_shape
    col_idx = node_idx % s[2]
    channel_idx = node_idx / (s[1]*s[2])
    row_idx = (node_idx % (s[1]*s[2])) / s[2]
    return (channel_idx, row_idx, col_idx)

def save2nifti(data, filename):
    """Save 3D data as nifti file.
    Original data shape is (18, 64, 64), and the resulting data shape is
    (64, 64, 18) which orientation is SRP."""
    # roll axis
    ndata = np.rollaxis(data, 0, 3)
    ndata = ndata[:, ::-1, :]
    # generate affine matrix
    aff = np.zeros((4, 4))
    aff[0, 1] = 2
    aff[1, 2] = -2.5
    aff[2, 0] = 2
    aff[3, 3] = 1
    img = nib.Nifti1Image(ndata, aff)
    nib.save(img, filename)

def plot_prf(prf_file):
    """Plot pRF."""
    prf_data = np.load(prf_file)
    vxl = prf_data[..., 0]
    # figure config

    for f in range(96):
        fig, axs = plt.subplots(5, 8)
        for t in range(40):
            tmp = vxl[:, t].reshape(96, 55, 55)
            tmp = tmp[f, :]
            im = axs[t/8][t%8].imshow(tmp, interpolation='nearest',
                                      cmap=plt.cm.ocean,
                                      vmin=-0.2, vmax=0.3)
        fig.colorbar(im)
        #plt.show()
        fig.savefig('%s.png'%(f))

def channel_sim(feat_file):
    """Compute similarity between each pair of channels."""
    feat = np.load(feat_file)
    print feat.shape
    feat = feat.reshape(96, 55, 55, 540)
    simmtx = np.zeros((feat.shape[0], feat.shape[0]))
    for i in range(feat.shape[0]):
        for j in range(i+1, feat.shape[0]):
            print '%s - %s' %(i, j)
            x = feat[i, :].reshape(-1, feat.shape[3])
            y = feat[j, :].reshape(-1, feat.shape[3])
            tmp = corr2_coef(x, y)
            tmp = tmp.diagonal()
            simmtx[i, j] = tmp.mean()
    np.save('sim_mtx.npy', simmtx)
    im = plt.imshow(simmtx, interpolation='nearest', cmap=plt.cm.ocean)
    plt.colorbar(im)
    plt.show()
 
