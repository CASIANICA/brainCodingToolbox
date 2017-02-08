# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy import ndimage
from scipy.misc import imsave

from brainDecTool.math import rcca
from sklearn.cross_decomposition import PLSCanonical

from brainDecTool.util import configParser
from brainDecTool.math import parallel_corr2_coef, corr2_coef
from brainDecTool.math import get_pls_components
from brainDecTool.math.norm import zero_one_norm
from brainDecTool.pipeline import retinotopy
from brainDecTool.pipeline.base import random_cross_modal_corr
from brainDecTool.pipeline.base import multiple_regression
from brainDecTool.pipeline.base import ridge_regression,random_ridge_regression
from brainDecTool.vim2 import util as vutil


def retinotopic_mapping(corr_file, mask=None):
    """Make the retinotopic mapping using activation map from CNN."""
    data_dir = os.path.dirname(corr_file)
    #fig_dir = os.path.join(data_dir, 'fig')
    #if not os.path.exists(fig_dir):
    #    os.mkdir(fig_dir, 0755)
    # load the cross-correlation matrix from file
    corr_mtx = np.load(corr_file, mmap_mode='r')
    #corr_mtx = np.memmap(corr_file, dtype='float16', mode='r',
    #                     shape=(73728, 3025))
    #                     shape=(73728, 290400))
    if isinstance(mask, np.ndarray):
        vxl_num = len(mask)
        vxl_idx = np.nonzero(mask==1)[0]
    else:
        vxl_num = corr_mtx.shape[0]
        vxl_idx = np.arange(vxl_num)
    pos_mtx = np.zeros((vxl_num, 2))
    pos_mtx[:] = np.nan
    for i in range(len(vxl_idx)):
        print 'Iter %s of %s' %(i, len(vxl_idx)),
        tmp = corr_mtx[i, :]
        tmp = np.nan_to_num(np.array(tmp))
        # significant threshold
        # one-tail test
        tmp[tmp <= 0.019257] = 0
        if np.sum(tmp):
            #tmp = tmp.reshape(55, 55)
            #mmtx = tmp
            tmp = tmp.reshape(96, 55, 55)
            mmtx = np.max(tmp, axis=0)
            print mmtx.min(), mmtx.max()
            #fig_file = os.path.join(fig_dir, 'v'+str(i)+'.png')
            #imsave(fig_file, mmtx)
            # get indices of n maximum values
            max_n = 20
            row_idx, col_idx = np.unravel_index(
                                        np.argsort(mmtx.ravel())[-1*max_n:],
                                        mmtx.shape)
            nmtx = np.zeros(mmtx.shape)
            nmtx[row_idx, col_idx] = mmtx[row_idx, col_idx]
            # center of mass
            x, y = ndimage.measurements.center_of_mass(nmtx)
            pos_mtx[vxl_idx[i], :] = [x, y]
        else:
            print ' '
    #receptive_field_file = os.path.join(data_dir, 'receptive_field_pos.npy')
    #np.save(receptive_field_file, pos_mtx)
    #pos_mtx = np.load(receptive_field_file)
    # eccentricity
    dist = retinotopy.coord2ecc(pos_mtx, (55, 55))
    # convert distance into degree
    # 0-4 degree -> d < 5.5
    # 4-8 degree -> d < 11
    # 8-12 degree -> d < 16.5
    # 12-16 degree -> d < 22
    # else > 16 degree
    ecc = np.zeros(dist.shape)
    for i in range(len(dist)):
        if np.isnan(dist[i]):
            ecc[i] = np.nan
        elif dist[i] < 5.445:
            ecc[i] = 1
        elif dist[i] < 10.91:
            ecc[i] = 2
        elif dist[i] < 16.39:
            ecc[i] = 3
        elif dist[i] < 21.92:
            ecc[i] = 4
        else:
            ecc[i] = 5
    #dist_vec = np.nan_to_num(ecc)
    #vol = dist_vec.reshape(18, 64, 64)
    vol = ecc.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max' + str(max_n) + '_ecc.nii.gz'))
    # angle
    angle_vec = retinotopy.coord2angle(pos_mtx, (55, 55))
    #angle_vec = np.nan_to_num(angle_vec)
    vol = angle_vec.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max'+ str(max_n) + '_angle.nii.gz'))

def ridge_retinotopic_mapping(corr_file, mask=None):
    """Make the retinotopic mapping using activation map from CNN."""
    data_dir = os.path.dirname(corr_file)
    #fig_dir = os.path.join(data_dir, 'fig')
    #if not os.path.exists(fig_dir):
    #    os.mkdir(fig_dir, 0755)
    # load the cross-correlation matrix from file
    corr_mtx = np.load(corr_file, mmap_mode='r')
    # corr_mtx.shape = (3025, 43007)
    corr_mtx = corr_mtx.T
    if isinstance(mask, np.ndarray):
        vxl_num = len(mask)
        vxl_idx = np.nonzero(mask==1)[0]
    else:
        vxl_num = corr_mtx.shape[0]
        vxl_idx = np.arange(vxl_num)
    pos_mtx = np.zeros((vxl_num, 2))
    pos_mtx[:] = np.nan
    for i in range(len(vxl_idx)):
        print 'Iter %s of %s' %(i, len(vxl_idx)),
        tmp = corr_mtx[i, :]
        tmp = np.nan_to_num(np.array(tmp))
        # significant threshold
        # one-tail test
        #tmp[tmp <= 0.019257] = 0
        if np.sum(tmp):
            tmp = tmp.reshape(55, 55)
            mmtx = tmp
            #tmp = tmp.reshape(96, 55, 55)
            #mmtx = np.max(tmp, axis=0)
            print mmtx.min(), mmtx.max()
            #fig_file = os.path.join(fig_dir, 'v'+str(i)+'.png')
            #imsave(fig_file, mmtx)
            # get indices of n maximum values
            max_n = 20
            row_idx, col_idx = np.unravel_index(
                                        np.argsort(mmtx.ravel())[-1*max_n:],
                                        mmtx.shape)
            nmtx = np.zeros(mmtx.shape)
            nmtx[row_idx, col_idx] = mmtx[row_idx, col_idx]
            # center of mass
            x, y = ndimage.measurements.center_of_mass(nmtx)
            pos_mtx[vxl_idx[i], :] = [x, y]
        else:
            print ' '
    #receptive_field_file = os.path.join(data_dir, 'receptive_field_pos.npy')
    #np.save(receptive_field_file, pos_mtx)
    #pos_mtx = np.load(receptive_field_file)
    # eccentricity
    dist = retinotopy.coord2ecc(pos_mtx, (55, 55))
    # convert distance into degree
    # 0-4 degree -> d < 5.5
    # 4-8 degree -> d < 11
    # 8-12 degree -> d < 16.5
    # 12-16 degree -> d < 22
    # else > 16 degree
    ecc = np.zeros(dist.shape)
    for i in range(len(dist)):
        if np.isnan(dist[i]):
            ecc[i] = np.nan
        elif dist[i] < 5.445:
            ecc[i] = 1
        elif dist[i] < 10.91:
            ecc[i] = 2
        elif dist[i] < 16.39:
            ecc[i] = 3
        elif dist[i] < 21.92:
            ecc[i] = 4
        else:
            ecc[i] = 5
    #dist_vec = np.nan_to_num(ecc)
    #vol = dist_vec.reshape(18, 64, 64)
    vol = ecc.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max' + str(max_n) + '_ecc.nii.gz'))
    # angle
    angle_vec = retinotopy.coord2angle(pos_mtx, (55, 55))
    #angle_vec = np.nan_to_num(angle_vec)
    vol = angle_vec.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max'+ str(max_n) + '_angle.nii.gz'))

def hrf_estimate(tf, feat_ts):
    """Estimate HRFs."""
    # voxel coordinates for test
    # voxels from R_V1
    # 0. (20, 36, 13) -> (20, 27, 13) -> (13, 20, 27)
    # 1. (20, 34, 13) -> (20, 29, 13) -> (13, 20, 29)
    # 2. (22, 34, 9) -> (22, 29, 9) -> (9, 22, 29)
    # voxels from L_V1
    # 3. (20, 29, 9) -> (20, 34, 9) -> (9, 20, 34)
    # 4. (16, 29, 12) -> (16, 34, 12) -> (12, 16, 34)

    voxels = [(20, 36, 13),
              (20, 34, 13),
              (22, 34, 9),
              (20, 29, 9),
              (16, 29, 12)]
    # voxel label validation
    #rv1_roi = tf.get_node('/roi/v1rh')
    #lv1_roi = tf.get_node('/roi/v1lh')
    # get time courses for each voxel
    vxl_idx = [vutil.coord2idx(coord) for coord in voxels]
    rt = tf.get_node('/rt')[:]
    vxl_data = rt[vxl_idx, :]
    vxl_data = np.nan_to_num(vxl_data)

    out = np.zeros((290400, 40, 5))
    for i in range(5):
        for j in range(feat_ts.shape[0]):
            print '%s - %s' %(i, j)
            tmp = feat_ts[j, :]
            tmp = (tmp - tmp.mean()) / tmp.std()
            out[j, :, i] = time_lag_corr(tmp, vxl_data[i, :], 40)
    np.save('hrf_test.npy', out)

def pls_y_pred_x(plsca, Y):
    """Predict X based on Y using a trained PLS CCA model `plsca`.
    """
    coef_ = np.dot(plsca.y_rotations_, plsca.x_loadings_.T)
    coef_ = (1./plsca.y_std_.reshape((plsca.y_weights_.shape[0], 1)) * coef_ *
            plsca.x_std_)
    # Normalize
    Yk = Y - plsca.y_mean_
    Yk /= plsca.y_std_
    Xpred = np.dot(Y, coef_)
    return Xpred + plsca.x_mean_

def plscorr_eval(train_fmri_ts, train_feat_ts, val_fmri_ts, val_feat_ts,
                 out_dir, mask_file):
    """Compute PLS correlation between brain activity and CNN activation."""
    train_feat_ts = train_feat_ts.reshape(-1, train_feat_ts.shape[3]).T
    val_feat_ts = val_feat_ts.reshape(-1, val_feat_ts.shape[3]).T
    train_fmri_ts = train_fmri_ts.T
    val_fmri_ts = val_fmri_ts.T

    # Iteration loop for different component number
    #for n in range(5, 19):
    #    print '--- Components number %s ---' %(n)
    #    plsca = PLSCanonical(n_components=n)
    #    plsca.fit(train_feat_ts, train_fmri_ts)
    #    pred_feat_c, pred_fmri_c = plsca.transform(val_feat_ts, val_fmri_ts)
    #    pred_fmri_ts = plsca.predict(val_feat_ts) 
    #    # calculate correlation coefficient between truth and prediction
    #    r = corr2_coef(val_fmri_ts.T, pred_fmri_ts.T, mode='pair')
    #    # get top 20% corrcoef for model evaluation
    #    vsample = int(np.rint(0.2*len(r)))
    #    print 'Sample size for evaluation : %s' % (vsample)
    #    r.sort()
    #    meanr = np.mean(r[-1*vsample:])
    #    print 'Mean prediction corrcoef : %s' %(meanr)
    
    # model generation based on optimized CC number
    cc_num = 10
    plsca = PLSCanonical(n_components=cc_num)
    plsca.fit(train_feat_ts, train_fmri_ts)
    from sklearn.externals import joblib
    joblib.dump(plsca, os.path.join(out_dir, 'plsca_model.pkl'))
    plsca = joblib.load(os.path.join(out_dir, 'plsca_model.pkl'))

    # calculate correlation coefficient between truth and prediction
    pred_fmri_ts = plsca.predict(val_feat_ts)
    fmri_pred_r = corr2_coef(val_fmri_ts.T, pred_fmri_ts.T, mode='pair')
    mask = vutil.data_swap(mask_file)
    vxl_idx = np.nonzero(mask.flatten()==1)[0]
    tmp = np.zeros_like(mask.flatten(), dtype=np.float64)
    tmp[vxl_idx] = fmri_pred_r
    tmp = tmp.reshape(mask.shape)
    vutil.save2nifti(tmp, os.path.join(out_dir, 'pred_fmri_r.nii.gz'))
    pred_feat_ts = pls_y_pred_x(plsca, val_fmri_ts)
    pred_feat_ts = pred_feat_ts.T.reshape(96, 14, 14, 540)
    np.save(os.path.join(out_dir, 'pred_feat.npy'), pred_feat_ts)

    # get PLS-CCA weights
    feat_cc, fmri_cc = plsca.transform(train_feat_ts, train_fmri_ts)
    np.save(os.path.join(out_dir, 'feat_cc.npy'), feat_cc)
    np.save(os.path.join(out_dir, 'fmri_cc.npy'), fmri_cc)
    feat_weight = plsca.x_weights_.reshape(96, 14, 14, cc_num)
    #feat_weight = plsca.x_weights_.reshape(96, 11, 11, cc_num)
    fmri_weight = plsca.y_weights_
    np.save(os.path.join(out_dir, 'feat_weights.npy'), feat_weight)
    np.save(os.path.join(out_dir, 'fmri_weights.npy'), fmri_weight)
    fmri_orig_ccs = get_pls_components(plsca.y_scores_, plsca.y_loadings_)
    np.save(os.path.join(out_dir, 'fmri_orig_ccs.npy'), fmri_orig_ccs)

def plscorr_viz(cca_dir, mask_file):
    """CCA weights visualization."""
    # plot feature weights (normalized)
    feat_weights = np.load(os.path.join(cca_dir, 'feat_weights.npy'))
    feat_weights = feat_weights.reshape(-1, 10)
    norm_feat_weights = zero_one_norm(feat_weights, two_side=True)
    norm_feat_weights = norm_feat_weights.reshape(96, 11, 11, 10)
    np.save(os.path.join(cca_dir, 'norm_feat_weights.npy'), norm_feat_weights)
    vutil.plot_cca_fweights(norm_feat_weights, cca_dir, 'norm2_feat_weight',
                            two_side=True)
    # plot fmri weights (normalized)
    fmri_weights = np.load(os.path.join(cca_dir, 'fmri_weights.npy'))
    norm_fmri_weights = zero_one_norm(fmri_weights, two_side=True)
    vutil.save_cca_volweights(norm_fmri_weights, mask_file, cca_dir,
                              'norm2_cca_weights', out_png=True, two_side=True)

    # show stimuli images corresponding to the largest fMRI activity
    fmri_cc = np.load(os.path.join(cca_dir, 'fmri_cc.npy'))
    for i in range(1, fmri_cc.shape[1]):
        print '------- CC #%s -------'%(i)
        tmp = fmri_cc[:, i].copy()
        print 'Negative side : index of top 10 images'
        print tmp.argsort()[:10]
        print 'Positive side : index of top 10 images'
        print tmp.argsort()[-10:]

    ## calculate corr between original variables and the CCs
    #feat_cc = np.load(os.path.join(out_dir, 'feat_cc.npy'))
    #parallel_corr2_coef(train_feat_ts.T, feat_cc.T, 
    #                    os.path.join(out_dir, 'feat_cc_corr.npy'),
    #                    block_size=10, n_jobs=1)
    #feat_cc_corr = np.load(os.path.join(out_dir, 'feat_cc_corr.npy'))
    #feat_cc_corr = feat_cc_corr.reshape(96, 11, 11, 10)
    #vutil.plot_cca_fweights(feat_cc_corr, out_dir, 'feat_cc_corr')
    ##vutil.fweights_bar(feat_cc_corr)
    #fmri_cc = np.load(os.path.join(out_dir, 'fmri_cc.npy'))
    #parallel_corr2_coef(train_fmri_ts.T, fmri_cc.T,
    #                    os.path.join(out_dir, 'fmri_cc_corr.npy'),
    #                    block_size=10, n_jobs=1)
    #fmri_cc_corr = np.load(os.path.join(out_dir, 'fmri_cc_corr.npy'))
    #vutil.save_cca_volweights(fmri_cc_corr, mask_file, out_dir,
    #                          prefix_name='fmri_cc_corr')

def inter_subj_cc_sim(subj1_id, subj2_id, subj_dir):
    """Compute inter-subjects CCs similarity."""
    subj1_dir = os.path.join(subj_dir, 'vS%s'%(subj1_id))
    subj2_dir = os.path.join(subj_dir, 'vS%s'%(subj2_id))
    #-- inter-channel similarity
    feat_weights_file1 = os.path.join(subj1_dir, 'plscca',
                                      'layer1', 'feat_weights.npy')
    feat_weights_file2 = os.path.join(subj2_dir, 'plscca',
                                      'layer1', 'feat_weights.npy')
    feat_cc_corr1 = np.load(feat_cc_corr_file1).reshape(96, 121, 10)
    feat_cc_corr2 = np.load(feat_cc_corr_file2).reshape(96, 121, 10)
    sim_mtx = np.zeros((960, 960))
    for i in  range(10):
        data1 = feat_cc_corr1[..., i]
        for j in range(10):
            data2 = feat_cc_corr2[..., j]
            tmp = corr2_coef(data1, data2)
            sim_mtx[i*96:(i+1)*96, j*96:(j+1)*96] = np.abs(tmp)
    np.save('feat_cc_weights_sim_subj_%s_%s.npy'%(subj1_id, subj2_id), sim_mtx)
    #-- inter-CC similarity
    #feat_cc_corr_file1 = os.path.join(subj1_dir, 'plscca',
    #                                  'layer1', 'feat_cc_corr.npy')
    #feat_cc_corr_file2 = os.path.join(subj2_dir, 'plscca',
    #                                  'layer1', 'feat_cc_corr.npy')
    #feat_cc_corr1 = np.load(feat_cc_corr_file1).reshape(96, 11, 11, 10)
    #feat_cc_corr2 = np.load(feat_cc_corr_file2).reshape(96, 11, 11, 10)
    #avg_weights1 = vutil.fweights_top_mean(feat_cc_corr1, 0.2)
    #avg_weights2 = vutil.fweights_top_mean(feat_cc_corr2, 0.2)
    #sim_mtx = corr2_coef(avg_weights1, avg_weights2)
    #np.save('feat_cc_sim_subj_%s_%s.npy'%(subj1_id, subj2_id), sim_mtx)

def reg_cca(train_fmri_ts, train_feat_ts, val_fmri_ts, val_feat_ts, out_dir):
    """Conduct CCA between brain activity and CNN activation."""
    train_feat_ts = train_feat_ts.reshape(-1, train_feat_ts.shape[3]).T
    val_feat_ts = val_feat_ts.reshape(-1, val_feat_ts.shape[3]).T
    train_fmri_ts = train_fmri_ts.T
    val_fmri_ts = val_fmri_ts.T

    #-- model training
    # for reduce complexity, a linear kernel is used
    #cca = rcca.CCACrossValidate(numCCs=[7, 8, 9, 10, 11, 12, 13],
    #                            kernelcca=True)
    CCnum = 7
    #cca = rcca.CCA(kernelcca=True, reg=0.007743, numCC=CCnum)
    #cca.train([train_feat_ts, train_fmri_ts])
    #cca.validate([val_feat_ts, val_fmri_ts])
    #cca.compute_ev([val_feat_ts, val_fmri_ts])
    #print 'Best CC number : %s' %(cca.best_numCC)
    #print 'Best reg : %s' %(cca.best_reg)
    out_file = os.path.join(out_dir, 'CCA_results_%s.hdf5'%(CCnum))
    #cca.save(os.path.join(out_file))
    
    #-- model exploring
    mask_file = r'/Users/sealhuang/brainDecoding/S1_mask.nii.gz'
    cca = rcca.CCA()
    cca.load(out_file)
    # model prediction performance
    fmri_pred_r = cca.corrs[1]
    feat_pred_r = cca.corrs[0].reshape(96, 11, 11)
    vutil.plot_cca_fweights(feat_pred_r, out_dir, 'pred_feat_r_CC%s'%(CCnum))
    mask = vutil.data_swap(mask_file)
    vxl_idx = np.nonzero(mask.flatten()==1)[0]
    tmp = np.zeros_like(mask.flatten(), dtype=np.float64)
    tmp[vxl_idx] = fmri_pred_r
    tmp = tmp.reshape(mask.shape)
    vutil.save2nifti(tmp, os.path.join(out_dir,
                     'pred_fmri_r_CC%s.nii.gz'%(CCnum)))

    # model weights visualization
    feat_weights = cca.ws[0]
    feat_weights = feat_weights.reshape(96, 11, 11, feat_weights.shape[1])
    fmri_weights = cca.ws[1]
    vutil.plot_cca_fweights(feat_weights, out_dir, 'feat_weight_CC%s'%(CCnum))
    vutil.save_cca_volweights(fmri_weights, mask_file, out_dir, 'cca_component')

    feat_cc = cca.comps[0]
    parallel_corr2_coef(train_feat_ts.T, feat_cc.T, 
                        os.path.join(out_dir, 'feat_cc_corr.npy'),
                        block_size=7, n_jobs=1)
    feat_cc_corr = np.load(os.path.join(out_dir, 'feat_cc_corr.npy'))
    feat_cc_corr = feat_cc_corr.reshape(96, 11, 11, 7)
    vutil.plot_cca_fweights(feat_cc_corr, out_dir, 'feat_cc_corr')

def roi_info(corr_mtx, wt_mtx, fmri_table, mask_idx, out_dir):
    """Get ROI info."""
    roi_list = ['v1lh', 'v1rh', 'v2lh', 'v2rh', 'v3lh', 'v3rh',
                'v3alh', 'v3arh', 'v3blh', 'v3brh', 'v4lh', 'v4rh']
    fingerprints = np.zeros((wt_mtx.shape[2], len(roi_list)))
    for ridx in range(len(roi_list)):
        roi_mask = fmri_table.get_node('/roi/%s'%(roi_list[ridx]))[:].flatten()
        roi_idx = np.nonzero(roi_mask==1)[0]
        roi_ptr = np.array([np.where(mask_idx==roi_idx[i])[0][0]
                            for i in range(len(roi_idx))])
        #-- plot pRF for each voxel
        roi_dir = os.path.join(out_dir, roi_list[ridx])
        os.system('mkdir %s'%(roi_dir))
        for idx in roi_ptr:
            tmp = corr_mtx[:, idx]
            if np.sum(tmp):
                tmp = tmp.reshape(55, 55)
                vutil.save_imshow(tmp, os.path.join(roi_dir,
                                                    '%s.png'%(mask_idx[idx])))
        #-- get feature response figure print
        ele_num = 0
        fp = np.zeros((fingerprints.shape[0]))
        for idx in roi_ptr:
            tmp = corr_mtx[:, idx]
            f = tmp>=0.17224
            if f.sum():
                ele_num += f.sum()
                fp += np.sum(wt_mtx[f, idx, :], axis=0)
        fp /= ele_num
        fingerprints[:, ridx] = fp
    np.save(os.path.join(out_dir, 'roi_fingerprints.npy'), fingerprints)

def permutation_stats(random_corr_mtx):
    """Get statistical estimate of `true` correlation coefficient."""
    vxl_num = random_corr_mtx.shape[2]
    maxv = random_corr_mtx.max(axis=0)
    for i in range(vxl_num):
        print maxv[:, i].max()
        print maxv[:, i].min()
        print '----------------'
    # get 95% corr coef across voxels
    maxv = maxv.flatten()
    maxv.sort()
    quar = maxv.shape[0]*0.95 - 1
    # 95% - 0.17224
    # 99% - 0.19019
    print maxv[int(np.rint(quar))]


if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures')
    db_dir = os.path.join(root_dir, 'subjects')

    # subj config
    subj_id = 1
    subj_dir = os.path.join(db_dir, 'vS%s'%(subj_id))
    
    #-- load fmri data
    #fmri_file = os.path.join(subj_dir, 'VoxelResponses.mat')
    #tf = tables.open_file(fmri_file)
    #tf.list_nodes
    #-- roi mat to nii
    #roi_file = os.path.join(subj_dir, 'S%s_small_roi.nii.gz'%(subj_id))
    #vutil.roi2nifti(tf, roi_file, mode='small')
    #-- get mean fmri responses
    #dataset = 'rt'
    #mean_file = os.path.join(subj_dir, 'S%s_mean_%s.nii.gz'%(subj_id, dataset))
    #vutil.gen_mean_vol(tf, dataset, mean_file)

    #-- load fmri response
    #train_fmri_ts = tf.get_node('/rt')[:]
    #val_fmri_ts = tf.get_node('/rv')[:]
    # data.shape = (73728, 540/7200)
    #-- load brain mask
    mask_file = os.path.join(subj_dir, 'S%s_mask.nii.gz'%(subj_id))
    mask = vutil.data_swap(mask_file).flatten()
    #vxl_idx = np.nonzero(mask==1)[0]
    #train_fmri_ts = np.nan_to_num(train_fmri_ts[vxl_idx])
    #val_fmri_ts = np.nan_to_num(val_fmri_ts[vxl_idx])
    
    #-- load cnn activation data
    #train_feat_file = os.path.join(feat_dir, 'conv1_train_trs.npy')
    #train_feat_ts = np.load(train_feat_file, mmap_mode='r')
    #val_feat_file = os.path.join(feat_dir, 'conv1_val_trs.npy')
    #val_feat_ts = np.load(val_feat_file, mmap_mode='r')
    # data.shape = (96, 55, 55, 540/7200)
    #-- load optical flow data: mag and ang
    #tr_mag_file = os.path.join(feat_dir, 'train_opticalflow_mag_trs_55_55.npy')
    #tr_mag_ts = np.load(tr_mag_file, mmap_mode='r')
    #val_mag_file = os.path.join(feat_dir, 'val_opticalflow_mag_trs_55_55.npy')
    #val_mag_ts = np.load(val_mag_file, mmap_mode='r')
    #tr_ang_file = os.path.join(feat_dir, 'train_opticalflow_ang_trs_55_55.npy')
    #tr_ang_ts = np.load(tr_ang_file, mmap_mode='r')
    #val_ang_file = os.path.join(feat_dir, 'val_opticalflow_ang_trs_55_55.npy')
    #val_ang_ts = np.load(val_ang_file, mmap_mode='r')
    # data.shape = (11, 11, 540/7200)
    
    #-- feature stack
    #print 'Feature stack ...'
    #train_feat_stack = np.vstack((train_feat_ts,
    #                              np.expand_dims(tr_mag_ts, axis=0),
    #                              np.expand_dims(tr_ang_ts, axis=0)))
    #val_feat_stack = np.vstack((val_feat_ts,
    #                            np.expand_dims(val_mag_ts, axis=0),
    #                            np.expand_dims(val_ang_ts, axis=0)))
    #tmp_train_file = os.path.join(feat_dir, 'train_conv1_optic_trs.npy')
    #np.save(tmp_train_file, train_feat_stack)
    #tmp_val_file = os.path.join(feat_dir, 'val_conv1_optic_trs.npy')
    #np.save(tmp_val_file, val_feat_stack)
    #del train_feat_ts, val_feat_ts, tr_mag_ts, tr_ang_ts, val_mag_ts
    #del val_ang_ts, train_feat_stack, val_feat_stack
    #train_feat_ts = np.load(tmp_train_file, mmap_mode='r')
    #val_feat_ts = np.load(tmp_val_file, mmap_mode='r')

    #-- calculate cross-modality corrlation 
    # sum up all channels
    # select parts of channels
    #feat_ts = feat_ts[0:48, :]
    #feat_ts = feat_ts.sum(axis=0)
    #retino_dir = os.path.join(subj_dir, 'retinotopic')
    #if not os.path.exists(retino_dir):
    #    os.mkdir(retino_dir, 0755)
    #corr_file = os.path.join(retino_dir, 'val_feat1_corr.npy')
    #val_feat_ts = val_feat_ts.reshape(290400, 540)
    #parallel_corr2_coef(val_fmri_ts, val_feat_ts, corr_file, block_size=96)
    #rand_corr_file = os.path.join(retino_dir, 'train_fmri_feat1_rand_corr.npy')
    #random_cross_modal_corr(fmri_ts, feat_ts, 10, 1000, rand_corr_file)
    #-- retinotopic mapping
    #retinotopic_mapping(corr_file, mask)
    
    #-- multiple regression voxel ~ channels from each location
    #regress_file = os.path.join(retino_dir, 'val_fmri_feat1_regress.npy')
    #roi_mask = get_roi_mask(tf)
    #multiple_regression(fmri_ts, feat_ts, regress_file)

    #-- ridge regression
    ridge_dir = os.path.join(subj_dir, 'ridge')
    if not os.path.exists(ridge_dir):
        os.mkdir(ridge_dir, 0755)
    #ridge_prefix = 'conv1_optical_pixel_wise_ridge'
    #ridge_regression(train_feat_ts, train_fmri_ts, val_feat_ts, val_fmri_ts,
    #                 ridge_dir, ridge_prefix, with_wt=True, n_cpus=4)
    #-- roi_stats
    corr_file = os.path.join(ridge_dir, 'conv1_optical_pixel_wise_corr.npy')
    wt_file = os.path.join(ridge_dir, 'conv1_optical_pixel_wise_weights.npy')
    #corr_mtx = np.load(corr_file, mmap_mode='r')
    #wt_mtx = np.load(wt_file, mmap_mode='r')
    #roi_info(corr_mtx, wt_mtx, tf, vxl_idx, ridge_dir)
    #-- retinotopic mapping
    ridge_retinotopic_mapping(corr_file, mask)
    #-- random regression
    #selected_vxl_idx = [5666, 9697, 5533, 5597, 5285, 5538, 5273, 5465, 38695,
    #                    38826, 42711, 46873, 30444, 34474, 38548, 42581, 5097,
    #                    5224, 5205, 9238, 9330, 13169, 17748, 21780]
    #train_fmri_ts = np.nan_to_num(train_fmri_ts[selected_vxl_idx])
    #val_fmri_ts = np.nan_to_num(val_fmri_ts[selected_vxl_idx])
    #print train_fmri_ts.shape
    #ridge_prefix = 'random_conv1_optical_pixel_wise'
    #random_ridge_regression(train_feat_ts, train_fmri_ts,
    #                        val_feat_ts, val_fmri_ts,
    #                        1000, ridge_dir, ridge_prefix)
    #-- permutation stats
    #rand_f = os.path.join(ridge_dir,'random_conv1_optical_pixel_wise_corr.npy')
    #random_corr_mtx = np.load(rand_f)
    #permutation_stats(random_corr_mtx)

    #-- PLS-CCA
    #pls_dir = os.path.join(subj_dir, 'plscca')
    #if not os.path.exists(pls_dir):
    #    os.mkdir(pls_dir, 0755)
    #cca_dir = os.path.join(pls_dir, 'layer1')
    #if not os.path.exists(cca_dir):
    #    os.mkdir(cca_dir, 0755)
    # combine layer1 features and optical flow features together
    #plscorr_eval(train_fmri_ts, train_feat_stack, val_fmri_ts, val_feat_stack,
    #             cca_dir, mask_file)
    #plscorr_eval(train_fmri_ts, train_feat_ts, val_fmri_ts, val_feat_ts,
    #             cca_dir, mask_file)
    #mask_file = os.path.join(subj_dir, 'S%s_mask.nii.gz'%(subj_id))
    #plscorr_viz(cca_dir, mask_file)
    #inter_subj_cc_sim(1, 2, db_dir)

    #-- regularized CCA
    #cca_dir = os.path.join(retino_dir, 'rcca', 'rcca_cc7')
    #if not os.path.exists(cca_dir):
    #    os.mkdir(cca_dir, 0755)
    #reg_cca(train_fmri_ts, train_feat_ts, val_fmri_ts, val_feat_ts, cca_dir)

    #-- close fmri data
    #tf.close()

