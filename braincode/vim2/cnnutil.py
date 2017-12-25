# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import matplotlib.pylab as plt
import nibabel as nib


def plot_pred_var(pls_dir, layer):
    train_r2 = np.load(os.path.join(pls_dir, '%s_pls_roi_r2.npy'%(layer)))
    val_r2 = np.load(os.path.join(pls_dir, '%s_pls_roi_r2_val.npy'%(layer)))
    maxv = max(train_r2.max(), val_r2.max())
    minv = min(train_r2.min(), val_r2.min())
    fig, axs = plt.subplots(1, 2)
    im = axs[0].imshow(train_r2, interpolation='none', vmax=maxv, vmin=minv)
    im = axs[1].imshow(val_r2, interpolation='none', vmax=maxv, vmin=minv)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    #fig.savefig(os.path.join(pls_dir, '%s_pred_r2.png'%(layer)))
    #plt.close()
    plt.show()

def assign_layer_index(pls_dir):
    layers = ['norm1', 'norm2', 'conv3', 'conv4',
              'pool5', 'fc6', 'fc7', 'fc8']
    merged_data = np.zeros((64, 18, 18, 8))
    for l in layers:
        nfile = os.path.join(pls_dir, '%s_optimal_pls_val_pred.nii.gz'%(layer))


