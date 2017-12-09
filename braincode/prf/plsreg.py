# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy import ndimage
from scipy.misc import imsave
from sklearn.cross_decomposition import PLSRegression
from sklearn.externals import joblib

from braincode.util import configParser
from braincode.math import parallel_corr2_coef, corr2_coef, ridge
from braincode.math import get_pls_components
from braincode.math.norm import zero_one_norm
from braincode.pipeline import retinotopy
from braincode.pipeline.base import random_cross_modal_corr
from braincode.vim2 import util as vutil
from braincode.prf import dataio


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
    # generate retinotopic mapping
    base_name = 'train_max' + str(max_n)
    prf2visual_angle(pos_mtx, img_size, data_dir, base_name)

def prf2visual_angle(prf_mtx, img_size, out_dir, base_name):
    """Generate retinotopic mapping based on voxels' pRF parameters.
    `prf_mtx` is a #voxel x pRF-features matrix, pRF features can be 2 columns
    (row, col) of image or 3 columns which adding a third pRF size parameters.

    """
    feature_size = prf_mtx.shape[1]
    pos_mtx = prf_mtx[:, :2]
    # eccentricity
    ecc = retinotopy.coord2ecc(pos_mtx, img_size, 20)
    vol = ecc.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(out_dir, base_name+'_ecc.nii.gz'))
    # angle
    angle = retinotopy.coord2angle(pos_mtx, img_size)
    vol = angle.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(out_dir, base_name+'_angle.nii.gz'))
    # pRF size
    if feature_size > 2:
        size_angle = retinotopy.get_prf_size(prf_mtx, 55, 20)
        vol = size_angle.reshape(18, 64, 64)
        vutil.save2nifti(vol, os.path.join(out_dir, base_name+'_size.nii.gz'))

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

def get_roi_idx(fmri_table, vxl_idx):
    """Get ROI label for each voxel."""
    rois = ['v1lh', 'v1rh', 'v2lh', 'v2rh', 'v3lh', 'v3rh', 'v3alh', 'v3arh',
            'v3blh', 'v3brh', 'v4lh', 'v4rh', 'MTlh', 'MTrh']
    roi_dict = {}
    for roi in rois:
        roi_mask = fmri_table.get_node('/roi/%s'%(roi))[:].flatten()
        roi_idx = np.nonzero(roi_mask==1)[0]
        roi_idx = np.intersect1d(roi_idx, vxl_idx)
        if roi_idx.sum():
            roi_ptr = np.array([np.where(vxl_idx==roi_idx[i])[0][0]
                                for i in range(len(roi_idx))])
            roi_dict[roi] = roi_ptr
    return roi_dict

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
    pass


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
    # directory config for database
    db_dir = cf.get('database', 'path')
    db_dir = os.path.join(db_dir, 'vim2')
    # directory config for analysis
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures', 'vim2')
    res_dir = os.path.join(root_dir, 'subjects')
 
    #-- general config config
    subj_id = 1
    subj_dir = os.path.join(res_dir, 'vim2_S%s'%(subj_id))
    pls_dir = os.path.join(subj_dir, 'pls')
    check_path(pls_dir)

    #-- load fmri data
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id)

    #-- load cnn activation data: data.shape = (feature_size, x, y, 7200/540)
    train_feat_file = os.path.join(feat_dir, 'norm1_train_trs.npy')
    train_feat_ts = np.load(train_feat_file, mmap_mode='r')
    val_feat_file = os.path.join(feat_dir, 'norm1_val_trs.npy')
    val_feat_ts = np.load(val_feat_file, mmap_mode='r')

    # PLS regression
    train_feat = train_feat_ts.reshape(-1, 7200).T
    train_fmri = train_fmri_ts.T
    print 'PLS model initializing ...'
    comps = 20
    pls2 = PLSRegression(n_components=comps)
    pls2.fit(train_feat, train_fmri)
    pred_train_fmri = pls2.predict(train_feat)
    pred_file = os.path.join(pls_dir, 'pls_pred_tfmri_c%s.npy'%(comps))
    np.save(pred_file, pred_train_fmri)
    joblib.dump(pls2, os.path.join(pls_dir, 'pls_model_c%s.pkl'%(comps)))
    for i in range(comps):
        print 'Component %s'%(i+1)
        print np.corrcoef(pls2.x_scores_[:, i], pls2.y_scores_[:, i])

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

    #-- Cross-modality mapping: voxel~CNN feature position correlation
    #cross_corr_dir = os.path.join(subj_dir, 'spatial_cross_corr')
    #check_path(cross_corr_dir)
    #-- features from CNN
    #corr_file = os.path.join(cross_corr_dir, 'train_conv1_corr.npy')
    #feat_ts = train_feat_ts.sum(axis=0).reshape(3025, 7200)
    #parallel_corr2_coef(train_fmri_ts, feat_ts, corr_file, block_size=55)
    #-- visual-pRF: select pixels which corr-coef greater than 1/2 maximum
    #corr_mtx = np.load(corr_file)
    #prf_dir = os.path.join(cross_corr_dir, 'prf')
    #visual_prf(corr_mtx, vxl_idx, prf_dir)
    #-- pRF stats and visualization for each ROI
    #prf_dir = os.path.join(cross_corr_dir, 'prf_figs')
    #check_path(prf_dir)
    #for roi in roi_dict:
    #    print '------%s------'%(roi)
    #    roi_idx = roi_dict[roi]
    #    # pRF type stats in each ROI
    #    roi_prf_type = prf_type[roi_idx]
    #    print 'Voxel number: %s'%(roi_prf_type.shape[0])
    #    for i in range(5):
    #        vxl_num = np.sum(roi_prf_type==(i+1))
    #        vxl_ratio = vxl_num * 100.0 / roi_prf_type.shape[0]
    #        print '%s, %0.2f'%(vxl_num, vxl_ratio)
    #    # save pRF as figs
    #    roi_dir = os.path.join(prf_dir, roi)
    #    check_path(roi_dir)
    #    roi_corr_mtx = corr_mtx[roi_idx, :]
    #    roi_min = roi_corr_mtx.min()
    #    roi_max = roi_corr_mtx.max()
    #    for i in roi_idx:
    #        vxl_prf = corr_mtx[i, :].reshape(55, 55)
    #        filename = 'v'+str(vxl_idx[i])+'_'+str(int(prf_type[i]))+'.png'
    #        out_file = os.path.join(roi_dir, filename)
    #        vutil.save_imshow(vxl_prf, out_file, val_range=(roi_min, roi_max))
    #-- curve-fit pRF visualization for each ROI
    #prf_dir = os.path.join(cross_corr_dir, 'fit_prf_figs')
    #check_path(prf_dir)
    #paras = np.load(os.path.join(cross_corr_dir, 'curve_fit_paras.npy'))
    #corr_mtx = np.load(os.path.join(cross_corr_dir, 'train_conv1_corr.npy'))
    #prf_type = np.load(os.path.join(cross_corr_dir, 'prf_type.npy'))
    #for roi in roi_dict:
    #    print '------%s------'%(roi)
    #    roi_idx = roi_dict[roi]
    #    # save pRF as figs
    #    roi_dir = os.path.join(prf_dir, roi)
    #    check_path(roi_dir)
    #    roi_corr_mtx = corr_mtx[roi_idx, :]
    #    roi_min = roi_corr_mtx.min()
    #    roi_max = roi_corr_mtx.max()
    #    for i in roi_idx:
    #        if np.isnan(paras[i, 0]):
    #            continue
    #        p = paras[i, :]
    #        vxl_prf = vutil.sugar_gaussian_f(55, *p).reshape(55, 55)
    #        filename = 'v'+str(vxl_idx[i])+'_'+str(int(prf_type[i]))+'.png'
    #        out_file = os.path.join(roi_dir, filename)
    #        vutil.save_imshow(vxl_prf, out_file, val_range=(roi_min, roi_max))
    #-- show pRF parameters on cortical surface
    #paras = np.load(os.path.join(cross_corr_dir, 'curve_fit_paras.npy'))
    #full_prf_mtx = np.zeros((73728, 3))
    #full_prf_mtx[:] = np.nan
    #for i in range(len(vxl_idx)):
    #    full_prf_mtx[vxl_idx[i], :] = paras[i, :3]
    #prf2visual_angle(full_prf_mtx, 55, cross_corr_dir, 'curve_fit')
    #err_file = os.path.join(cross_corr_dir, 'curve_fit_err.nii.gz')
    #vutil.vxl_data2nifti(paras[:, 5], vxl_idx, err_file)

    #-- Cross-modality mapping: voxel~CNN unit correlation
    #cross_corr_dir = os.path.join(subj_dir, 'cross_corr')
    #check_path(cross_corr_dir)
    # features from CNN
    #corr_file = os.path.join(cross_corr_dir, 'train_norm1_corr.npy')
    #feat_ts = train_feat_ts.reshape(69984, 7200)
    #parallel_corr2_coef(train_fmri_ts, feat_ts, corr_file, block_size=96)
    #-- random cross-modal correlation
    #rand_corr_file = os.path.join(cross_corr_dir, 'rand_train_conv1_corr.npy')
    #permutation_stats(np.load(rand_corr_file))

    #-- fmri data z-score
    #print 'fmri data temporal z-score'
    #m = np.mean(train_fmri_ts, axis=1, keepdims=True)
    #s = np.std(train_fmri_ts, axis=1, keepdims=True)
    #train_fmri_ts = (train_fmri_ts - m) / (1e-10 + s)
    #m = np.mean(val_fmri_ts, axis=1, keepdims=True)
    #s = np.std(val_fmri_ts, axis=1, keepdims=True)
    #val_fmri_ts = (val_fmri_ts - m) / (1e-10 + s)

