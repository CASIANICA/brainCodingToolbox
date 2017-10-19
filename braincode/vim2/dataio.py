# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
import nibabel as nib

from braincode.util import configParser

def data_swap(nifti_file):                                                      
    """Convert nifti data into original data shape."""
    data = nib.load(nifti_file).get_data()
    ndata = data[:, ::-1, :]
    ndata = np.rollaxis(ndata, 0, 3)
    ndata = np.rollaxis(ndata, 0, 3)
    return ndata

def load_fmri(subj_dir, roi=None):
    """Load fmri time courses for each voxel within specified ROI."""
    fmri_file = os.path.join(subj_dir, 'VoxelResponses.mat')
    tf = tables.open_file(fmri_file)
    # create mask
    # train data shape: (73728, 7200)
    train_ts = tf.get_node('/rt')[:]
    # get non-NaN voxel index
    fmri_s = train_ts.sum(axis=1)
    non_nan_idx = np.nonzero(np.logical_not(np.isnan(fmri_s)))[0]
    if roi:
        roi_mask = tf.get_node('/roi/%s'%(roi))[:].flatten()
        vxl_idx = np.nonzero(roi_mask==1)[0]
    else:
        full_mask_file = os.path.join(subj_dir, 'func_mask.nii.gz')
        full_mask = data_swap(full_mask_file).flatten()
        vxl_idx = np.nonzero(full_mask==1)[0]
    vxl_idx = np.intersect1d(vxl_idx, non_nan_idx)
    # load fmri response
    # data shape: (#voxel, 7200/540)
    train_ts = tf.get_node('/rt')[:]
    train_ts = np.nan_to_num(train_ts[vxl_idx])
    val_ts = tf.get_node('/rv')[:]
    val_ts = np.nan_to_num(val_ts[vxl_idx])
    return vxl_idx, train_ts, val_ts


