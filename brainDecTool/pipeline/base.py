# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from joblib import Parallel, delayed
from brainDecTool.math import corr2_coef

def cross_modal_corr(fmri_ts, feat_ts, filename, block_size=32):
    """Compute cross-modality correlation between fMRI response and image
    features derived from CNN.
    
    Usage
    -----
    cross_modal_corr(fmri_ts, feat_ts, filename, memmap=False, block_size=32)
    
    Return
    ------
    A cross-modality correlation matrix saved in `filename`. For example, if
    the size of `fmri_ts` is (p, n), and the size of `feat_ts` is (q, n), the
    size of return matrix is (p, q).

    Note
    ----
    'block_size' : the number of rows in `feat_ts` processed in one iter.
    """
    # to reduce memory usage, we compute Pearson's r iteratively
    fmri_size = fmri_ts.shape[0]
    feat_size = feat_ts.shape[0]
    corr_mtx = np.memmap(filename, dtype='float16', mode='w+',
                         shape=(fmri_size, feat_size))
    print 'Compute cross-modality correlation ...'
    # parallelize the corr computation
    Parallel(n_jobs=5)(delayed(f)(fmri_ts, feat_ts, corr_mtx, i, block_size)
                                  for i in range(feat_size/block_size))

def f(in_fmri, in_feat, output, i, block_size):
    """Sugar function for parallel computing."""
    print 'Iter %s' %(i)
    output[:, i*block_size:(i+1)*block_size] = corr2_coef(in_fmri,
                in_feat[i*block_size:(i+1)*block_size, :])

