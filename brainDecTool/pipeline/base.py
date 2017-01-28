# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from joblib import Parallel, delayed
from brainDecTool.math import corr2_coef, ols_fit, ridge


def random_cross_modal_corr(fmri_ts, feat_ts, voxel_num, iter_num, filename):
    """Generate a random distribution of correlation corfficient."""
    corr_mtx = np.memmap(filename, dtype='float16', mode='w+',
                         shape=(voxel_num, iter_num))
    print 'Compute cross-modality correlation ...'
    fmri_size = fmri_ts.shape[0]
    feat_size = feat_ts.shape[0]
    # select voxels and features randomly
    vxl_idx = np.random.choice(fmri_size, voxel_num, replace=False)
    feat_idx = np.random.choice(feat_size, voxel_num, replace=False)
    for i in range(voxel_num):
        print 'voxel index %s' %(vxl_idx[i])
        print 'feature index %s' %(feat_idx[i])
        feat_data = feat_ts[feat_idx[i], :].reshape(1, -1)
        fmri_data = np.zeros((iter_num, fmri_ts.shape[1]))
        for j in range(iter_num):
            fmri_data[j, :] = np.random.permutation(fmri_ts[vxl_idx[i]])
        corr_mtx[i, :] = corr2_coef(feat_data, fmri_data)

def multiple_regression(fmri_ts, feat_ts, filename):
    """Multiple regression between voxel time course and channels from each
    location."""
    fmri_size = fmri_ts.shape[0]
    feat_size = feat_ts.shape
    reg_mtx = np.memmap(filename, dtype='float16', mode='w+',
                        shape=(fmri_size, feat_size[1], feat_size[2]))
    print 'Compute multiple regression correlation ...'
    Parallel(n_jobs=4)(delayed(mrf)(fmri_ts, feat_ts, reg_mtx, v)
                                    for v in range(fmri_size))
    
    narray = np.array(reg_mtx)
    np.save(filename, narray)

def mrf(in_fmri, in_feat, out, idx):
    """Sugar function for multiple regression."""
    print idx
    y = in_fmri[idx, :]
    feat_size = in_feat.shape
    for i in range(feat_size[1]):
        for j in range(feat_size[2]):
            #print '%s-%s-%s' %(idx, i, j)
            x = in_feat[:, i, j, :].T
            out[idx, i, j] = ols_fit(y, x)

def ridge_regression(train_feat, train_fmri, val_feat, val_fmri,
                     out_dir, prefix):
    """Calculate ridge regression between features from one pixel location and
    the fmri responses from all voxels.
    """
    feat_size = train_feat.shape[0]
    pixel_size = (train_feat.shape[1], train_feat.shape[2])
    voxel_size = train_fmri.shape[0]
    corr_file = os.path.join(out_dir, prefix+'_corr.npy')
    wt_file = os.path.join(out_dir, prefix+'_weights.npy')
    corr_mtx = np.memmap(corr_file, dtype='float16', mode='w+',
                         shape=(pixel_size[0]*pixel_size[1], voxel_size))
    wt_mtx = np.memmap(wt_file, dtype='float16', mode='w+',
                    shape=(pixel_size[0]*pixel_size[1], voxel_size, feat_size))
    print 'Compute Ridge regreesion for each pixel ...'
    Parallel(n_jobs=4)(delayed(ridge_sugar)(train_feat, train_fmri, val_feat,
                                            val_fmri, corr_mtx, wt_mtx, v)
                                            for v in [(i, j)
                                                for i in range(pixel_size[0])
                                                for j in range(pixel_size[1])])
    narray = np.array(corr_mtx)
    np.save(corr_file, narray)
    narray = np.array(wt_mtx)
    np.save(wt_file, narray)

def ridge_sugar(train_feat, train_fmri, val_feat, val_fmri,
                corr_mtx, wt_mtx, idx):
    """Sugar function for ridge regression."""
    pixel_size = (train_feat.shape[1], train_feat.shape[2])
    row, col = idx[0], idx[1]
    print 'row %s - col %s' % (row, col)
    tfeat = train_feat[:, row, col, :]
    m = np.mean(tfeat)
    s = np.std(tfeat)
    tfeat = (tfeat-m) / (1e-10+s)
    vfeat = (val_feat[:, row, col, :]-m) / (1e-10+s)
    wt, corr, valphas, bscores, valinds = ridge.bootstrap_ridge(tfeat.T, train_fmri.T, vfeat.T, val_fmri.T, alphas=np.logspace(-2, 2, 20), nboots=5, chunklen=100, nchunks=10, single_alpha=True)
    corr_mtx[row*pixel_size[0]+col] = corr
    wt_mtx[row*pixel_size[0]+col] = wt.T

