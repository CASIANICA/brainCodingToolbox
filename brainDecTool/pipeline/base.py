# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from brainDecTool.math import corr2_coef

def cross_modal_corr(fmri_ts, feat_ts, filename, memmap=False):
    """Compute cross-modality correlation between fMRI response and image
    features derived from CNN.
    
    Usage
    -----
    cross_modal_corr(fmri_ts, feat_ts, filename, memmap=False)
    
    Return
    ------
    A cross-modality correlation matrix saved in `filename`. For example, if
    the size of `fmri_ts` is (p, n), and the size of `feat_ts` is (q, n), the
    size of return matrix is (p, q).

    Note
    ----
    `memmap` is set as False in default, if True, a memmap would be used to
    save memory.
    """
    # to reduce memory usage, we compute Pearson's r iteratively
    fmri_size = fmri_ts.shape[0]
    feat_size = feat_ts.shape[0]
    if memmap:
        corr_mtx = np.memmap(filename, dtype='float16', mode='w+',
                             shape=(fmri_size, feat_size))
    else:
        corr_mtx = np.zeros((fmri_size, feat_size), dtype=np.float16)
    print 'Compute cross-modality correlation ...'
    for i in range(feat_size):
        print 'Iter %s of %s' %(i, feat_size)
        tmp = feat_ts[i, :].reshape(1, -1)
        corr_mtx[:, i] = corr2_coef(fmri_ts, tmp)[:, 0]
    if memmap:
        del corrmtx
    else:
        np.save(filename, corr_mtx)

