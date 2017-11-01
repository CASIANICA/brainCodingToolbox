# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
import seaborn as sns

def load_prf_data(roi_dir):
    """Load all pRF data for specific ROI."""                                   
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
    vxl_num = sel_vxl.shape[0]
    gabor_corr = np.load(os.path.join(roi_dir, 'gabor_contributes.npy'))
    #gabor_r2 = np.square(gabor_corr)
    df_dict = {}
    for i in range(5):
        #x = np.sum(orig_data['paras'][:, (i*8):(i*8+8)], axis=1)
        #df_dict['wt_sum_%s'%(i+1)] = x[sel_vxl]
        df_dict['wt_sum_%s'%(i+1)] = gabor_corr[sel_vxl, i]
        #df_dict['wt_sum_%s'%(i+1)] = gabor_r2[sel_vxl, i]
    df_dict['ecc'] = orig_data['ecc'][sel_vxl]
    df = pd.DataFrame(df_dict)

    return orig_data, df

def plot_gabor_contrib(roi, df):
    """Plot tunning contribution of each gabor sub-banks VS. ecc."""
    sns.set(color_codes=True)
    for i in range(5):
        g = sns.lmplot(x='ecc', y='wt_sum_%s'%(i+1), data=df, x_bins=10,
                       size=6, order=2)
        g.set(xlim=(0, 14), ylim=(-0.3, 0.8))
        g.savefig('gabor_contrib_%s_sp%s.png'%(roi, i+1))

