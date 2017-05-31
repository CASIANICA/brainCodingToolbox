# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import numpy as np
import tables
from scipy import ndimage
from scipy.misc import imsave
from sklearn.cross_decomposition import PLSCanonical

from brainDecTool.util import configParser
from brainDecTool.math import parallel_corr2_coef, corr2_coef, ridge
from brainDecTool.math import get_pls_components, rcca
from brainDecTool.math import LinearRegression
from brainDecTool.math.norm import zero_one_norm
from brainDecTool.pipeline import retinotopy
from brainDecTool.pipeline.base import random_cross_modal_corr
from brainDecTool.vim2 import util as vutil


def check_path(dir_path):
    """Check whether the directory does exist, if not, create it."""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path, 0755)

def retinotopic_mapping(corr_file, data_dir, vxl_idx=None, figout=False):
    """Make the retinotopic mapping using activation map from CNN."""
    if figout:
        fig_dir = os.path.join(data_dir, 'fig')
        check_path(fig_dir)
    # load the cross-correlation matrix from file
    corr_mtx = np.load(corr_file, mmap_mode='r')
    # set voxel index
    if not isinstance(vxl_idx, np.ndarray):
        vxl_idx = np.arange(corr_mtx.shape[0])
    elif len(vxl_idx) != corr_mtx.shape[0]:
        print 'mismatch on voxel number!'
        return
    else:
        print 'voxel index loaded.'
    img_size = 55.0
    pos_mtx = np.zeros((73728, 2))
    pos_mtx[:] = np.nan
    for i in range(len(vxl_idx)):
        print 'Iter %s of %s' %(i+1, len(vxl_idx)),
        tmp = corr_mtx[i, :]
        tmp = np.nan_to_num(np.array(tmp))
        # significant threshold for one-tail test
        tmp[tmp <= 0.019257] = 0
        if np.sum(tmp):
            mmtx = tmp.reshape(55, 55)
            #tmp = tmp.reshape(96, 27, 27)
            #mmtx = np.max(tmp, axis=0)
            print mmtx.min(), mmtx.max()
            if figout:
                fig_file = os.path.join(fig_dir, 'v'+str(vxl_idx[i])+'.png')
                imsave(fig_file, mmtx)
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
    dist = retinotopy.coord2ecc(pos_mtx, (img_size, img_size))
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
        elif dist[i] < img_size/10:
            ecc[i] = 1
        elif dist[i] < img_size/5:
            ecc[i] = 2
        elif dist[i] < img_size/10*3:
            ecc[i] = 3
        elif dist[i] < img_size/10*4:
            ecc[i] = 4
        else:
            ecc[i] = 5
    #dist_vec = np.nan_to_num(ecc)
    #vol = dist_vec.reshape(18, 64, 64)
    vol = ecc.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max' + str(max_n) + '_ecc.nii.gz'))
    # angle
    angle_vec = retinotopy.coord2angle(pos_mtx, (img_size, img_size))
    #angle_vec = np.nan_to_num(angle_vec)
    vol = angle_vec.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max'+ str(max_n) + '_angle.nii.gz'))

def visual_prf(corr_mtx, vxl_idx, prf_dir):
    """pRF visualization."""
    check_path(prf_dir)
    prf = np.zeros_like(corr_mtx)
    for i in range(len(vxl_idx)):
        orig_mtx = corr_mtx[i, :].reshape(55, 55)
        orig_file = os.path.join(prf_dir, 'v'+str(vxl_idx[i])+'_orig.png')
        imsave(orig_file, orig_mtx)
        prf_mtx = orig_mtx.copy()
        prf_mtx[prf_mtx<prf_mtx.max()*0.8] = 0
        prf_file = os.path.join(prf_dir, 'v'+str(vxl_idx[i])+'_prf.png')
        imsave(prf_file, prf_mtx)
        prf[i, :] = prf_mtx.flatten()
    np.save(os.path.join(prf_dir, 'prf.npy'), prf)

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
                'v3alh', 'v3arh', 'v3blh', 'v3brh', 'v4lh', 'v4rh',
                'MTlh', 'MTrh', 'MTplh', 'MTprh']
    fingerprints = np.zeros((wt_mtx.shape[2], len(roi_list)))
    for ridx in range(len(roi_list)):
        roi_mask = fmri_table.get_node('/roi/%s'%(roi_list[ridx]))[:].flatten()
        roi_idx = np.nonzero(roi_mask==1)[0]
        roi_idx = np.intersect1d(roi_idx, mask_idx)
        roi_ptr = np.array([np.where(mask_idx==roi_idx[i])[0][0]
                            for i in range(len(roi_idx))])
        #-- plot pRF for each voxel
        roi_dir = os.path.join(out_dir, roi_list[ridx])
        os.system('mkdir %s'%(roi_dir))
        for idx in roi_ptr:
            tmp = corr_mtx[:, idx]
            if np.sum(tmp):
                tmp = tmp.reshape(13, 13)
                vutil.save_imshow(tmp, os.path.join(roi_dir,
                                                    '%s.png'%(mask_idx[idx])))
            else:
                print 'Drop %s'%(idx)
        #-- get feature response figure print
        ele_num = 0
        fp = np.zeros((fingerprints.shape[0]))
        for idx in roi_ptr:
            tmp = corr_mtx[:, idx]
            # conv1+optical : 0.17419
            # norm1 : 0.15906
            # norm2 : 0.14636
            # conv3 : 0.14502
            f = tmp>=0.14502
            if f.sum():
                ele_num += f.sum()
                fp += np.sum(wt_mtx[f, idx, :], axis=0)
        fp /= ele_num
        fingerprints[:, ridx] = fp
    #-- plot fingerprint for each roi
    #for i in range(len(roi_list)):
    #    plt.bar(np.arange(96), fingerprints[:96, i], 0.35)
    #    plt.savefig('%s.png'%(roi_list[i]))
    #    plt.close()
    np.save(os.path.join(out_dir, 'roi_fingerprints.npy'), fingerprints)

def permutation_stats(random_corr_mtx):
    """Get statistical estimate of `true` correlation coefficient."""
    #vxl_num = random_corr_mtx.shape[2]
    #for i in range(vxl_num):
    #    print maxv[:, i].max()
    #    print maxv[:, i].min()
    #    print '----------------'
    maxv = random_corr_mtx.max(axis=1)
    # get 95% corr coef across voxels
    #maxv = maxv.flatten()
    maxv.sort()
    quar = maxv.shape[0]*0.95 - 1
    # 95% - 0.17224
    # 99% - 0.19019
    print 'Correlation threshold for permutation test: ',
    print maxv[int(np.rint(quar))]


if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures')
    db_dir = os.path.join(root_dir, 'subjects')
 
    # phrase 'test': analyses were only conducted within lV1 for code test
    # phrase 'work': for real analyses
    phrase = 'test'
 
    # subj config
    subj_id = 2
    subj_dir = os.path.join(db_dir, 'vS%s'%(subj_id))
 
    #-- load fmri data
    fmri_file = os.path.join(subj_dir, 'VoxelResponses.mat')
    tf = tables.open_file(fmri_file)
    #tf.list_nodes
    #-- roi mat to nii
    #roi_file = os.path.join(subj_dir, 'S%s_small_roi.nii.gz'%(subj_id))
    #vutil.roi2nifti(tf, roi_file, mode='small')
    #-- get mean fmri responses
    #dataset = 'rt'
    #mean_file = os.path.join(subj_dir, 'S%s_mean_%s.nii.gz'%(subj_id, dataset))
    #vutil.gen_mean_vol(tf, dataset, mean_file)

    #-- create mask
    train_fmri_ts = tf.get_node('/rt')[:]
    # data.shape = (73728, 7200)
    # get non-nan voxel indexs
    fmri_s = train_fmri_ts.sum(axis=1)
    non_nan_idx = np.nonzero(np.logical_not(np.isnan(fmri_s)))[0]
    # create mask
    full_mask_file = os.path.join(subj_dir, 'S%s_mask.nii.gz'%(subj_id))
    full_mask = vutil.data_swap(full_mask_file).flatten()
    full_vxl_idx = np.nonzero(full_mask==1)[0]
    full_vxl_idx = np.intersect1d(full_vxl_idx, non_nan_idx)
    if phrase=='test':
        mask_file = os.path.join(subj_dir, 'S%s_small_roi.nii.gz'%(subj_id))
        mask = vutil.data_swap(mask_file).flatten()
        mask[mask>1] = 0
        mask[mask>0] = 1
        vxl_idx = np.nonzero(mask==1)[0]
        vxl_idx = np.intersect1d(vxl_idx, non_nan_idx)
    else:
        vxl_idx = full_vxl_idx

    #-- load fmri response
    # data shape: (selected_voxel, 7200/540)
    train_fmri_ts = tf.get_node('/rt')[:]
    train_fmri_ts = np.nan_to_num(train_fmri_ts[vxl_idx])
    #val_fmri_ts = tf.get_node('/rv')[:]
    #val_fmri_ts = np.nan_to_num(val_fmri_ts[vxl_idx])
    ##-- save masked data as npy file
    #train_file = os.path.join(subj_dir, 'S%s_train_fmri_lV1.npy'%(subj_id))
    #val_file = os.path.join(subj_dir, 'S%s_val_fmri_lV1.npy'%(subj_id))
    #np.save(train_file, train_fmri_ts)
    #np.save(val_file, val_fmri_ts)

    #-- load cnn activation data
    # data.shape = (feature_size, x, y, 7200/540)
    train_feat_file = os.path.join(feat_dir, 'conv1_train_trs.npy')
    train_feat_ts = np.load(train_feat_file, mmap_mode='r')
    #val_feat_file = os.path.join(feat_dir, 'conv1_val_trs.npy')
    #val_feat_ts = np.load(val_feat_file, mmap_mode='r')

    #-- load salience data
    #train_sal_file = os.path.join(feat_dir, 'salience_train_55_55_trs.npy')
    #train_sal_ts = np.load(train_sal_file, mmap_mode='r')
    #val_sal_file = os.path.join(feat_dir, 'salience_val_55_55_trs.npy')
    #val_sal_ts = np.load(val_sal_file, mmap_mode='r')

    #-- load salience-modulated cnn features
    #train_salfeat_file = os.path.join(feat_dir, 'conv1_train_trs_salmod.npy')
    #train_salfeat_ts = np.load(train_salfeat_file, mmap_mode='r')
    #val_salfeat_file = os.path.join(feat_dir, 'conv1_val_trs_salmod.npy')
    #val_salfeat_ts = np.load(val_salfeat_file, mmap_mode='r')

    #-- 2d gaussian kernel based pRF estimate
    prf_dir = os.path.join(subj_dir, 'prf')
    check_path(prf_dir)
    gaussian_prf_file = os.path.join(feat_dir, 'gaussian_prfs.npy')
    gaussian_prfs = np.load(gaussian_prf_file, mmap_mode='r')
    prf_num = gaussian_prfs.shape[2]
    fwhms = np.arange(1, 11)
    # feat processing
    train_feat_ts = train_feat_ts.mean(axis=0)
    train_feat_ts = train_feat_ts.reshape(3025, 7200)
    vxl_prf = np.zeros((len(vxl_idx), 3))
    for i in range(len(vxl_idx)):
        print 'Voxel %s of %s'%(i+1, len(vxl_idx))
        vxl_ts = train_feat_ts[i, :]
        corr_tmp = np.zeros(prf_num)
        for j in range(prf_num):
            prf_tmp = gaussian_prfs[..., j]
            feat_ts = prf_tmp.reshape(3025, ).T.dot(train_feat_ts)
            corr_tmp[j] = np.corrcoef(vxl_ts, feat_ts)[0][1]
        max_idx = np.argmax(corr_tmp)
        vxl_prf[i, 0] = max_idx % (55*prf_num) / prf_num
        vxl_prf[i, 1] = max_idx / (55*prf_num)
        vxl_prf[i, 2] = max_idx % prf_num
    np.save(os.path.join(prf_dir, 'lv1_vxl_prf.npy'), vxl_prf)

    #-- feature temporal z-score
    #print 'CNN features temporal z-score ...'
    ## summary features across channels
    #train_feat_ts = train_feat_ts.mean(axis=0)
    #train_feat_m = train_feat_ts.mean(axis=2, keepdims=True)
    #train_feat_s = train_feat_ts.std(axis=2, keepdims=True)
    #train_feat_ts = (train_feat_ts-train_feat_m)/(1e-10+train_feat_s)
    #val_feat_ts = val_feat_ts.mean(axis=0)
    #val_feat_m = val_feat_ts.mean(axis=2, keepdims=True)
    #val_feat_s = val_feat_ts.std(axis=2, keepdims=True)
    #val_feat_ts = (val_feat_ts-val_feat_m)/(1e-10+val_feat_s)
    #print 'Salience features temporal z-score ...'
    #train_sal_m = train_sal_ts.mean(axis=2, keepdims=True)
    #train_sal_s = train_sal_ts.std(axis=2, keepdims=True)
    #train_sal_ts = (train_sal_ts-train_sal_m)/(1e-10+train_sal_s)
    #val_sal_m = val_sal_ts.mean(axis=2, keepdims=True)
    #val_sal_s = val_sal_ts.std(axis=2, keepdims=True)
    #val_sal_ts = (val_sal_ts-val_sal_m)/(1e-10+val_sal_s)
    #print 'Salience modulated features temporal z-score ...'
    #train_salfeat_ts = train_salfeat_ts.mean(axis=0)
    #train_salfeat_m = train_salfeat_ts.mean(axis=2, keepdims=True)
    #train_salfeat_s = train_salfeat_ts.std(axis=2, keepdims=True)
    #train_salfeat_ts=(train_salfeat_ts-train_salfeat_m)/(1e-10+train_salfeat_s)
    #val_salfeat_ts = val_salfeat_ts.mean(axis=0)
    #val_salfeat_m = val_salfeat_ts.mean(axis=2, keepdims=True)
    #val_salfeat_s = val_salfeat_ts.std(axis=2, keepdims=True)
    #val_salfeat_ts = (val_salfeat_ts-val_salfeat_m)/(1e-10+val_salfeat_s)

    #-- voxel-wise linear regression
    #cross_corr_dir = os.path.join(subj_dir, 'spatial_cross_corr', 'lv1')
    #reg_dir = os.path.join(cross_corr_dir, 'linreg')
    #check_path(reg_dir)
    #corr_mtx = np.load(os.path.join(cross_corr_dir, 'train_conv1_corr.npy'))
    #corr_mtx = corr_mtx.reshape(470, 55, 55)
    ## voxel-wise linear regression
    #wts = np.zeros((470, 55, 55, 3))
    #train_corr = np.zeros((470, 55, 55))
    #val_corr = np.zeros((470, 55, 55))
    #wts_mask = np.zeros((470, 3))
    #statsp_mask = np.zeros((470, 3))
    #train_corr_mask = np.zeros(470,)
    #val_corr_mask = np.zeros(470, )
    #for i in range(len(vxl_idx)):
    #    print 'Voxel %s of %s ...'%(i+1, len(vxl_idx))
    #    prf = corr_mtx[i, ...].copy()
    #    prf = prf > prf.max()*0.8
    #    print '%s voxels selected'%(prf.sum())
    #    if not prf.sum():
    #        continue
    #    pos = np.nonzero(prf)
    #    wts_tmp = np.zeros((pos[0].shape[0], 3))
    #    statsp_tmp = np.zeros((pos[0].shape[0], 3))
    #    train_corr_tmp = np.zeros(pos[0].shape[0],)
    #    val_corr_tmp = np.zeros(pos[0].shape[0],)
    #    for j in range(pos[0].shape[0]):
    #        train_Y = train_fmri_ts[i, :]
    #        val_Y = val_fmri_ts[i, :]
    #        train_X = np.zeros((7200, 3))
    #        train_X[:, 0] = train_feat_ts[pos[0][j], pos[1][j], :]
    #        train_X[:, 1] = train_sal_ts[pos[0][j], pos[1][j], :]
    #        train_X[:, 2] = train_salfeat_ts[pos[0][j], pos[1][j], :]
    #        val_X = np.zeros((540, 3))
    #        val_X[:, 0] = val_feat_ts[pos[0][j], pos[1][j], :]
    #        val_X[:, 1] = val_sal_ts[pos[0][j], pos[1][j], :]
    #        val_X[:, 2] = val_salfeat_ts[pos[0][j], pos[1][j], :]
    #        model = LinearRegression(fit_intercept=False)
    #        model.fit(train_X, train_Y)
    #        wts[i, pos[0][j], pos[1][j], :] = model.coef_
    #        ptrain_Y = model.predict(train_X)
    #        tcorr = np.corrcoef(ptrain_Y, train_Y)[0][1]
    #        train_corr[i, pos[0][j], pos[1][j]] = tcorr
    #        pval_Y = model.predict(val_X)
    #        vcorr = np.corrcoef(pval_Y, val_Y)[0][1]
    #        val_corr[i, pos[0][j], pos[1][j]] = vcorr
    #        wts_tmp[j, :] = model.coef_
    #        statsp_tmp[j, :] = model.p
    #        train_corr_tmp[j] = tcorr
    #        val_corr_tmp[j] = vcorr
    #    wts_mask[i, :] = wts_tmp.mean(axis=0)
    #    statsp_mask[i, :] = statsp_tmp.mean(axis=0)
    #    train_corr_mask[i] = train_corr_tmp.mean()
    #    val_corr_mask[i] = val_corr_tmp.mean()
    #np.save(os.path.join(reg_dir, 'wts.npy'), wts)
    #np.save(os.path.join(reg_dir, 'train_corr.npy'), train_corr)
    #np.save(os.path.join(reg_dir, 'val_corr.npy'), val_corr)
    #np.save(os.path.join(reg_dir, 'wts_mask.npy'), wts_mask)
    #np.save(os.path.join(reg_dir, 'stats_p_mask.npy'), statsp_mask)
    #np.save(os.path.join(reg_dir, 'train_corr_mask.npy'), train_corr_mask)
    #np.save(os.path.join(reg_dir, 'val_corr_mask.npy'), val_corr_mask)

    #-- Cross-modality mapping: voxel~CNN feature position correlation
    #cross_corr_dir = os.path.join(subj_dir, 'spatial_cross_corr', 'lv1')
    #check_path(cross_corr_dir)
    #-- features from CNN
    #corr_file = os.path.join(cross_corr_dir, 'train_conv1_corr.npy')
    #feat_ts = train_feat_ts.sum(axis=0)
    #feat_ts = feat_ts.reshape(3025, 7200)
    #parallel_corr2_coef(train_fmri_ts, feat_ts, corr_file, block_size=55)
    #-- pRF visualization
    #-- select pixels which cross-modality greater than 1/2 maximum
    #corr_mtx = np.load(corr_file)
    #prf_dir = os.path.join(cross_corr_dir, 'prf')
    #visual_prf(corr_mtx, vxl_idx, prf_dir)

    #-- Cross-modality mapping: voxel~CNN unit correlation
    #cross_corr_dir = os.path.join(subj_dir, 'cross_corr')
    #check_path(cross_corr_dir)
    # features from CNN
    #corr_file = os.path.join(cross_corr_dir, 'train_norm1_corr.npy')
    #feat_ts = train_feat_ts.reshape(69984, 7200)
    #parallel_corr2_coef(train_fmri_ts, feat_ts, corr_file, block_size=96)
    # features from optical flow
    #corr_file = os.path.join(cross_corr_dir, 'train_optic_mag_corr.npy')
    #feat_ts = tr_mag_ts.reshape(16384, 7200)
    #parallel_corr2_coef(train_fmri_ts, feat_ts, corr_file, block_size=55)
    #-- random cross-modal correlation
    #rand_corr_file = os.path.join(cross_corr_dir, 'rand_train_conv1_corr.npy')
    #feat_ts = tr_mag_ts.reshape(16384, 7200)
    #random_cross_modal_corr(train_fmri_ts, feat_ts, 1000, 1000, rand_corr_file)
    #permutation_stats(np.load(rand_corr_file))
 
    #-- retinotopic mapping based on cross-correlation with norm1
    #cross_corr_dir = os.path.join(subj_dir, 'cross_corr')
    #retino_dir = os.path.join(cross_corr_dir, 'retinotopic')
    #check_path(retino_dir)
    #corr_file = os.path.join(cross_corr_dir, 'train_norm1_corr.npy')
    #retinotopic_mapping(corr_file, retino_dir, vxl_idx, figout=False)

    #-- cnn layer assignment based on cross-correlation
    #cross_corr_dir = os.path.join(subj_dir, 'cross_corr')
    #layer_names = ['norm1', 'norm2', 'conv3', 'conv4', 'pool5']
    #vxl_num = len(vxl_idx)
    #layer_num = len(layer_names)
    #max_corr = np.zeros((vxl_num, layer_num))
    #for i in range(layer_num):
    #    l = layer_names[i]
    #    corr_file = os.path.join(cross_corr_dir, 'train_%s_corr.npy'%l)
    #    corr = np.nan_to_num(np.load(corr_file))
    #    max_corr[:, i] = corr.max(axis=1)
    #max_corr_file = os.path.join(cross_corr_dir, 'max_corr_across_layers.npy')
    #np.save(max_corr_file, max_corr)
    #layer_idx = np.argmax(max_corr, axis=1) + 1
    #layer_file = os.path.join(cross_corr_dir, 'layer_mapping.nii.gz')
    #vutil.vxl_data2nifti(layer_idx, vxl_idx, layer_file)

    #-- feature temporal z-score
    #print 'CNN features temporal z-score ...'
    #train_feat_m = train_feat_ts.mean(axis=3, keepdims=True)
    #train_feat_s = train_feat_ts.std(axis=3, keepdims=True)
    #train_feat_ts = (train_feat_ts-train_feat_m)/(1e-10+train_feat_s)
    #val_feat_ts = (val_feat_ts-train_feat_m)/(1e-10+train_feat_s)
    #tmp_train_file = os.path.join(feat_dir, 'train_conv1_trs_z.npy')
    #np.save(tmp_train_file, train_feat_ts)
    #del train_feat_ts
    #tmp_val_file = os.path.join(feat_dir, 'val_norm1_trs_z.npy')
    #np.save(tmp_val_file, val_feat_ts)
    #del val_feat_ts
    #train_feat_ts = np.load(tmp_train_file, mmap_mode='r')
    #train_feat_ts = train_feat_ts.reshape(69984, 7200)
    #val_feat_ts = np.load(tmp_val_file, mmap_mode='r')
    #val_feat_ts = val_feat_ts.reshape(69984, 540)

    #-- fmri data z-score
    #print 'fmri data temporal z-score'
    #m = np.mean(train_fmri_ts, axis=1, keepdims=True)
    #s = np.std(train_fmri_ts, axis=1, keepdims=True)
    #train_fmri_ts = (train_fmri_ts - m) / (1e-10 + s)
    #m = np.mean(val_fmri_ts, axis=1, keepdims=True)
    #s = np.std(val_fmri_ts, axis=1, keepdims=True)
    #val_fmri_ts = (val_fmri_ts - m) / (1e-10 + s)

    #-- Encoding: ridge regression
    #ridge_dir = os.path.join(subj_dir, 'ridge')
    #check_path(ridge_dir)
    #-- layer-wise ridge regression: select cnn units whose correlation with
    #-- the given voxel exceeded the half of the maximal correlation within
    #-- the layer.
    #cross_corr_dir = os.path.join(subj_dir, 'cross_corr')
    #cross_corr_file = os.path.join(cross_corr_dir, 'train_norm1_corr.npy')
    #cross_corr = np.load(cross_corr_file, mmap_mode='r')
    ## output config
    #ALPHA_NUM = 20
    #BOOTS_NUM = 15
    #full_vxl_num, feat_num = cross_corr.shape
    #vxl_num = len(vxl_idx)
    #wt_mtx = np.zeros((vxl_num, feat_num))
    #alpha_mtx = np.zeros(vxl_num)
    #val_corr_mtx = np.zeros(vxl_num)
    ##bootstrap_corr_mtx = np.zeros((vxl_num, ALPHA_NUM, BOOTS_NUM))
    #bootstrap_corr_mtx = np.zeros((vxl_num, BOOTS_NUM))
    ## voxel-wise regression
    #for i in range(vxl_num):
    #    print 'Voxel %s in %s'%(i+1, vxl_num)
    #    v_corr = cross_corr[np.where(full_vxl_idx==vxl_idx[i])[0][0], :]
    #    feat_idx = v_corr > (v_corr.max()/2)
    #    print 'Select %s features'%(feat_idx.sum())
    #    vtrain_feat = train_feat_ts[feat_idx, :]
    #    vval_feat = val_feat_ts[feat_idx, :]
    #    vtrain_fmri = np.expand_dims(train_fmri_ts[i, :], axis=0)
    #    vval_fmri = np.expand_dims(val_fmri_ts[i, :], axis=0)
    #    wt, val_corr, alpha, bscores, valinds = ridge.bootstrap_ridge(
    #                            vtrain_feat.T, vtrain_fmri.T,
    #                            vval_feat.T, vval_fmri.T,
    #                            alphas=np.arange(100, 2001, 2001/ALPHA_NUM),
    #                            #alphas=np.logspace(-2, 3, ALPHA_NUM),
    #                            nboots=BOOTS_NUM, chunklen=72, nchunks=20,
    #                            single_alpha=False, use_corr=True)
    #    print 'Alpha: %s'%(alpha)
    #    print 'Val Corr: %s'%(val_corr)
    #    wt_mtx[i, feat_idx] = wt.T
    #    val_corr_mtx[i] = val_corr
    #    alpha_mtx[i] = alpha
    #    alpha_idx = np.where(np.arange(100, 2001, 2001/ALPHA_NUM)==alpha)[0][0]
    #    #alpha_idx = np.where(np.logspace(-2, 3, ALPHA_NUM)==alpha)[0][0]
    #    bootstrap_corr_mtx[i, :] = bscores[alpha_idx, 0, :]
    #    #bootstrap_corr_mtx[i, ...] = bscores[:, 0, :]
    ## save output
    #wt_file = os.path.join(ridge_dir, 'norm1_wt.npy')
    #alpha_file = os.path.join(ridge_dir, 'norm1_alpha.npy')
    #val_corr_file = os.path.join(ridge_dir, 'norm1_val_corr.npy')
    #bootstrap_corr_file = os.path.join(ridge_dir, 'norm1_bootstrap_corr.npy')
    #np.save(wt_file, wt_mtx)
    #np.save(alpha_file, alpha_mtx)
    #np.save(val_corr_file, val_corr_mtx)
    #np.save(bootstrap_corr_file, bootstrap_corr_mtx)

    #-- assign layer index based on CV accuracy
    #layer_names = ['norm1', 'norm2', 'conv3', 'conv4', 'pool5']
    #vxl_num = len(vxl_idx)
    #layer_num = len(layer_names)
    #cv_acc = np.zeros((vxl_num, layer_num))
    #for i in range(layer_num):
    #    l = layer_names[i]
    #    corr_file = os.path.join(ridge_dir, '%s_bootstrap_corr.npy'%l)
    #    cv_acc[:, i] = np.load(corr_file).mean(axis=1)
    #cv_acc_file = os.path.join(ridge_dir, 'max_corr_across_layers.npy')
    #np.save(cv_acc_file, cv_acc)
    #layer_idx = np.argmax(cv_acc, axis=1) + 1
    #layer_file = os.path.join(ridge_dir, 'layer_mapping.nii.gz')
    #vutil.vxl_data2nifti(layer_idx, vxl_idx, layer_file)

    #-- visualizing cortical representation of each voxel
    #v_idx = 74
    #wt_file = os.path.join(ridge_dir, 'norm1_wt.npy')
    #wt = np.load(wt_file, mmap_mode='r')
    ## rescale weight
    #print train_feat_s.max(), train_feat_s.min()
    #wt = wt[v_idx, :] * train_feat_s.reshape(69984,)
    ## reshape val_feat_ts
    #feat_ts = val_feat_ts.reshape(69984, 540)
    #pred_ts = np.zeros_like(feat_ts)
    #for i in range(feat_ts.shape[1]):
    #    pred_ts[:, i] = np.array(np.power(feat_ts[:, i]+1e-10, wt) - 1)
    #pred_ts = np.nan_to_num(pred_ts.reshape(96, 27, 27, 540))
    #pred_file = os.path.join(ridge_dir, 'vxl_%s_pred_norm1.npy'%(v_idx))
    #np.save(pred_file, pred_ts)

