# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pandas as pd

import seaborn as sns

def load_roi_prf(roi_dir):
    """Load all pRF data for specific ROI.
    Usage:
        orig_data, df = load_roi_prf(roi_dir)

        # plot tunning contribution of different subset of gabor banks
        import matplotlib.pylab as plt
        import seaborn as sns
        import numpy as np

        orig_data, df = load_roi_prf('...gaussian_kernel/v1lh')
        sns.set(color_codes=True)
        sns.lmplot(x='ecc', y='corr', hue='frequency', col='frequency',
                   data=df, order=2, x_bins=np.arange(1, 11, 1))
        plt.show()

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
    # select significantly predicted voxels
    thr = 0.17
    sel_vxl = orig_data['val_corr']>=thr
    # get tunning contribution of different spatial frequencies
    gabor_corr = np.load(os.path.join(roi_dir, 'gabor_contributes.npy'))
    df_dict = {}
    df_dict['corr'] = gabor_corr[sel_vxl, :].flatten()
    df_dict['frequency'] = [1, 2, 3, 4, 5] * sel_vxl.sum()
    df_dict['ecc'] = np.repeat(orig_data['ecc'][sel_vxl], 5)
    df = pd.DataFrame(df_dict)

    return orig_data, df

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
        ecc = np.load(roi_dir, 'ecc.npy')
        gabor_corr = np.load(roi_dir, 'gabor_contributes.npy')
        val_corr = np.load(roi_dir, 'reg_sel_model_corr.npy')
        # select significantly predicted voxels
        thr = 0.17
        sel_vxl = val_corr>=thr
        hemi_dict = {'corr': gabor_corr[sel_vxl, :].flatten(),
                     'frequenct': [1, 2, 3, 4, 5] * sel_vxl.sum(),
                     'ecc': np.repeat(ecc[sel_vxl], 5)}
        df_list.append(pd.DataFrame(hemi_dict))
    if not np.sum(hemi_flags):
        print 'No roi found'
        return
    # plot
    sns.set(color_codes=True)
    if lh_merged and np.sum(hemi_flags)==2:
        df = pd.concat(df_list)
        g = sns.lmplot(x='ecc', y='corr', hue='frequency', col='frequency',
                       data=df, order=2, x_bins=np.arange(1, 11, 1))
        g.savefig('%s_gabor_contrib.png'%(roi))
    else:
        if np.sum(hemi_flags)==2:
            for df in df_list:
                g = sns.lmplot(x='ecc', y='corr', hue='frequency',
                               col='frequency', data=df, order=2,
                               x_bins=np.arange(1, 11, 1))
                g.savefig('%s_gabor_contrib.png'%(roi+hemis[df_list.index(df)]))
        else:
            g = sns.lmplot(x='ecc', y='corr', hue='frequency', col='frequency',
                           data=df, order=2, x_bins=np.arange(1, 11, 1))
            g.savefig('%s_gabor_contrib.png'%(roi+hemis[hemi_flags.index(1)]))

def load_gabor_contrib(prf_dir):
    """Load tunning contribution of different gabor sub-banks."""
    rois = ['v1lh', 'v2lh', 'v3lh', 'v4lh']
    df = None
    ecc_range_min = [0, 2, 4, 6, 8]
    ecc_range_max = [2, 4, 6, 8, 14]
    for i in range(5):
        corr_vtr = np.empty(0)
        freq_idx = []
        roi_vtr = []
        for roi in rois:
            val_corr = np.load(os.path.join(prf_dir, roi,
                                'reg_sel_model_corr.npy'))
            ecc = np.load(os.path.join(prf_dir, roi, 'ecc.npy'))
            gabor_corr = np.load(os.path.join(prf_dir, roi,
                                'gabor_contributes.npy'))
            sig_vxl = val_corr>=0.17
            emin_vxl = ecc>ecc_range_min[i]
            emax_vxl = ecc<=ecc_range_max[i]
            sel_vxl = emin_vxl * emax_vxl * sig_vxl
            tmp_corr = gabor_corr[sel_vxl, :]
            if roi_vtr:
                freq_idx = freq_idx + [1, 2, 3, 4, 5] * tmp_corr.shape[0]
                tmp_corr = tmp_corr.flatten()
                roi_vtr = roi_vtr + [roi] * tmp_corr.shape[0]
                corr_vtr = np.concatenate((corr_vtr, tmp_corr))
            else:
                freq_idx = [1, 2, 3, 4, 5] * tmp_corr.shape[0]
                corr_vtr = tmp_corr.flatten()
                roi_vtr = [roi] * corr_vtr.shape[0]
        if isinstance(df, pd.DataFrame):
            df_dict = {'ecc_%s_freq_idx'%(i+1): np.array(freq_idx),
                       'ecc_%s_corr'%(i+1): corr_vtr,
                       'ecc_%s_roi'%(i+1): roi_vtr}
            tmp_df = pd.DataFrame(df_dict)
            df = pd.concat([df, tmp_df], axis=1)
        else:
            df_dict = {'ecc_%s_freq_idx'%(i+1): np.array(freq_idx),
                       'ecc_%s_corr'%(i+1): corr_vtr,
                       'ecc_%s_roi'%(i+1): roi_vtr}
            df = pd.DataFrame(df_dict)
    return df


