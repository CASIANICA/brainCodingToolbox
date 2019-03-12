# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pandas as pd

import seaborn as sns

def load_roi_prf(roi_dir):
    """Load all pRF data for specific ROI.
    Usage:
        orig_data = load_roi_prf(roi_dir)
    """                                   
    orig_data = {}
    # load estimated data
    data_type = {'angle': 'angle',
                 'ecc': 'ecc',
                 'sel_model_idx': 'reg_sel_model',
                 'val_corr': 'reg_sel_model_corr',
                 'paras': 'reg_sel_paras'}
    for item in data_type.keys():
        dfile = os.path.join(roi_dir, data_type[item]+'.npy')
        orig_data[item] = np.load(dfile)

    return orig_data

def plot_gabor_contrib(prf_dir, roi, lhmerged=True):
    """Plot tunning contribution of each gabor sub-banks VS. ecc."""
    df_list = []
    hemis = ['lh', 'rh']
    hemi_flags = [1, 1]
    for hemi in hemis:
        roi_dir = os.path.join(prf_dir, roi+hemi)
        if not os.path.exists(roi_dir):
            print '%s data does not exists'%(roi+hemi)
            hemi_flags[hemis.index(hemi)] = 0
            continue
        # if roi_dir exists, load prf data
        df_list.append(load_gabor_contrib(roi_dir))
        #ecc = np.load(os.path.join(roi_dir, 'ecc.npy'))
        #gabor_corr = np.load(os.path.join(roi_dir, 'gabor_contributes.npy'))
        #val_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
        ## select significantly predicted voxels
        #thr = 0.17
        #sel_vxl = val_corr>=thr
        #hemi_dict = {'corr': gabor_corr[sel_vxl, :].flatten(),
        #             'frequency': [1, 2, 3, 4, 5] * sel_vxl.sum(),
        #             'ecc': np.repeat(ecc[sel_vxl], 5)}
        #df_list.append(pd.DataFrame(hemi_dict))
    if not np.sum(hemi_flags):
        print 'No roi found'
        return
    # plot
    sns.set(color_codes=True)
    if lhmerged and np.sum(hemi_flags)==2:
        df = pd.concat(df_list)
        g = sns.lmplot(x='ecc', y='corr', hue='frequency', col='frequency',
                       data=df, order=2, x_bins=np.arange(1, 11, 1))
        g.savefig('%s_gabor_contrib.png'%(roi))
    else:
        if np.sum(hemi_flags)==2:
            for i in range(2):
                g = sns.lmplot(x='ecc', y='corr', hue='frequency',
                               col='frequency', data=df_list[i], order=2,
                               x_bins=np.arange(1, 11, 1))
                g.savefig('%s_gabor_contrib.png'%(roi+hemis[i]))
        else:
            g = sns.lmplot(x='ecc', y='corr', hue='frequency', col='frequency',
                           data=df, order=2, x_bins=np.arange(1, 11, 1))
            g.savefig('%s_gabor_contrib.png'%(roi+hemis[hemi_flags.index(1)]))

def load_gabor_contrib(roi_dir):
    """Load tunning contribution of different gabor sub-banks."""
    ecc = np.load(os.path.join(roi_dir, 'ecc.npy'))
    gabor_corr = np.load(os.path.join(roi_dir, 'gabor_contributes.npy'))
    val_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    # select significantly predicted voxels
    thr = 0.17
    sel_vxl = val_corr>=thr
    hemi_dict = {'corr': gabor_corr[sel_vxl, :].flatten(),
                 'frequency': [1, 2, 3, 4, 5] * sel_vxl.sum(),
                 'ecc': np.repeat(ecc[sel_vxl], 5)}
    return pd.DataFrame(hemi_dict)

def vim1_plot_gabor_contrib(prf_dir, roi):
    """Plot tunning contribution of each gabor sub-banks VS. ecc."""
    roi_dir = os.path.join(prf_dir, roi)
    if not os.path.exists(roi_dir):
        print '%s data does not exists'%(roi)
        return
    # if roi_dir exists, load prf data
    df = vim1_load_gabor_contrib(roi_dir)
    # plot
    sns.set(color_codes=True)
    g = sns.lmplot(x='ecc', y='corr', hue='frequency', col='frequency',
                   data=df, order=2, x_bins=np.arange(1, 11, 1))
    axes = g.axes
    axes[0, 0].set_xlim(0,)
    axes[0, 0].set_ylim(-0.4, 1.0)
    g.savefig('%s_gabor_contrib.png'%(roi))

def vim1_load_gabor_contrib(roi_dir):
    """Load tunning contribution of different gabor sub-banks."""
    ecc = np.load(os.path.join(roi_dir, 'ecc.npy'))
    gabor_corr = np.load(os.path.join(roi_dir, 'gabor_contributes.npy'))
    val_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    # select significantly predicted voxels
    thr = 0.28
    sel_vxl = val_corr>=thr
    print 'Selected %s voxels'%(sel_vxl.shape[0])
    hemi_dict = {'corr': gabor_corr[sel_vxl, :].flatten(),
                 'frequency': [1, 2, 3, 4, 5, 6, 7, 8, 9] * sel_vxl.sum(),
                 'ecc': np.repeat(ecc[sel_vxl], 9)}
    return pd.DataFrame(hemi_dict)

