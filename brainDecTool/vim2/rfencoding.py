# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy import ndimage
from scipy.misc import imsave
import scipy.optimize as opt
from sklearn.cross_decomposition import PLSCanonical
from sklearn.linear_model import LassoCV

from brainDecTool.util import configParser
from brainDecTool.math import parallel_corr2_coef, corr2_coef, ridge
from brainDecTool.math import get_pls_components, rcca
from brainDecTool.math import LinearRegression
from brainDecTool.math.norm import zero_one_norm, zscore
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
    subj_id = 1
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
    if phrase=='test':
        lv1_mask = tf.get_node('/roi/v1lh')[:].flatten()
        vxl_idx = np.nonzero(lv1_mask==1)[0]
        # for vS1, lV1 contains 490 non-NaN voxels
        vxl_idx = np.intersect1d(vxl_idx, non_nan_idx)
    else:
        full_mask_file = os.path.join(subj_dir, 'S%s_mask.nii.gz'%(subj_id))
        full_mask = vutil.data_swap(full_mask_file).flatten()
        full_vxl_idx = np.nonzero(full_mask==1)[0]
        vxl_idx = np.intersect1d(full_vxl_idx, non_nan_idx)
        #np.save(os.path.join(subj_dir, 'full_vxl_idx.npy'), vxl_idx)
    roi_dict = get_roi_idx(tf, vxl_idx)
    #np.save(os.path.join(subj_dir, 'roi_idx_pointer.npy'), roi_dict)
    #roi_dict = np.load(os.path.join(subj_dir, 'roi_idx_pointer.npy')).item()

    #-- load fmri response
    # data shape: (#voxel, 7200/540)
    train_fmri_ts = tf.get_node('/rt')[:]
    train_fmri_ts = np.nan_to_num(train_fmri_ts[vxl_idx])
    val_fmri_ts = tf.get_node('/rv')[:]
    val_fmri_ts = np.nan_to_num(val_fmri_ts[vxl_idx])
    #-- save masked data as npy file
    #train_file = os.path.join(subj_dir, 'S%s_train_fmri_lV1.npy'%(subj_id))
    #val_file = os.path.join(subj_dir, 'S%s_val_fmri_lV1.npy'%(subj_id))
    #np.save(train_file, train_fmri_ts)
    #np.save(val_file, val_fmri_ts)

    #-- load cnn activation data
    # data.shape = (feature_size, x, y, 7200/540)
    #train_feat_file = os.path.join(feat_dir, 'conv1_train_trs.npy')
    #train_feat_ts = np.load(train_feat_file, mmap_mode='r')
    #val_feat_file = os.path.join(feat_dir, 'conv1_val_trs.npy')
    #val_feat_ts = np.load(val_feat_file, mmap_mode='r')

    #-- 2d gaussian kernel based pRF estimate
    prf_dir = os.path.join(subj_dir, 'prf')
    check_path(prf_dir)
    # parameter config
    fwhms = np.arange(1, 11)
    # lasso linear regression
    vxl_idx = vxl_idx[:10]
    file_idx = -1
    for i in range(30250):
        print '--------------------------'
        print 'Kernel %s'%(i+1)
        # load CNN features modulated by Gaussian kernels
        if i/550 > file_idx:
            train_feat_file = os.path.join(feat_dir, 'gaussian_kernels',
                                    'gaussian_conv1_train_trs_%s.npy'%(i/550))
            train_feat_ts = np.load(train_feat_file)
            val_feat_file = os.path.join(feat_dir, 'gaussian_kernels',
                                    'gaussian_conv1_val_trs_%s.npy'%(i/550))
            val_feat_ts = np.load(val_feat_file)
            file_idx = i/550
        train_x = train_feat_ts[..., i%550]
        val_x = val_feat_ts[..., i%550]
        # shape of x : (96, 7200/540)
        train_x = zscore(train_x).T
        val_x = zscore(val_x).T
        # output vars
        paras = np.zeros((96, 30250, len(vxl_idx)))
        val_corr = np.zeros((30250, len(vxl_idx)))
        alphas = np.zeros((30250, len(vxl_idx)))
        for j in range(len(vxl_idx)):
            print 'Voxel %s'%(j+1)
            train_y = train_fmri_ts[j]
            val_y = val_fmri_ts[j]
            lasso_cv = LassoCV(cv=10, n_jobs=4)
            lasso_cv.fit(train_x, train_y)
            alphas[i, j] = lasso_cv.alpha_
            paras[:, i, j] = lasso_cv.coef_
            pred_y = lasso_cv.predict(val_x)
            val_corr[i, j] = np.corrcoef(val_y, pred_y)[0][1]
            print 'Alpha %s, prediction score %s'%(alphas[i, j], val_corr[i, j])
    np.save(os.path.join(prf_dir, 'lassoreg_paras.npy'), paras)
    np.save(os.path.join(prf_dir, 'lassoreg_pred_corr.npy'), val_corr)
    np.save(os.path.join(prf_dir, 'lassoreg_alphas.npy'), alphas)

    #-- pRF to retinotopy
    #prf_mtx = np.load(os.path.join(prf_dir, 'vxl_prf.npy'))
    ## generate full voxel feature matrix
    #full_prf_mtx = np.zeros((73728, 3))
    #full_prf_mtx[:] = np.nan
    #for i in range(len(vxl_idx)):
    #    full_prf_mtx[vxl_idx[i], :] = prf_mtx[i, :]
    #prf2visual_angle(full_prf_mtx, 55, prf_dir, 'retinotopy')

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
    #reg_dir = os.path.join(cross_corr_dir, 'linreg_l1')
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
    #-- categorize voxels based on pRF types
    #corr_file = os.path.join(cross_corr_dir, 'train_conv1_corr.npy')
    #corr_mtx = np.load(corr_file)
    ## get pRF by remove non-significant pixels
    ## two-tailed p < 0.01: r > 0.0302 and r < -0.0302
    #ncorr_mtx = corr_mtx.copy()
    #ncorr_mtx[(corr_mtx<=0.0302)&(corr_mtx>=-0.0302)] = 0
    #prf_max = ncorr_mtx.max(axis=1)
    #prf_min = ncorr_mtx.min(axis=1)
    #prf_type = np.zeros(corr_mtx.shape[0])
    #prf_type[(prf_max>0)&(prf_min>0)] = 1
    #prf_type[(prf_max>0)&(prf_min==0)] = 2
    #prf_type[(prf_max>0)&(prf_min<0)] = 3
    #prf_type[(prf_max==0)&(prf_min<0)] = 4
    #prf_type[(prf_max<0)&(prf_min<0)] = 5
    #np.save(os.path.join(cross_corr_dir, 'prf_type.npy'), prf_type)
    #nii_file = os.path.join(cross_corr_dir, 'prf_type.nii.gz')
    #vutil.vxl_data2nifti(prf_type, vxl_idx, nii_file)
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
    #-- get pRF parameters based on 2D Gaussian curve using model fitting
    #corr_mtx = np.load(os.path.join(cross_corr_dir, 'train_conv1_corr.npy'))
    ## last column is curve fitting error based on squared-differnece
    #paras = np.zeros((corr_mtx.shape[0], 6))
    #for i in range(corr_mtx.shape[0]):
    #    print i,
    #    y = corr_mtx[i, :]
    #    if y.max() >= abs(y.min()):
    #        x0, y0 = np.unravel_index(np.argmax(y.reshape(55, 55)), (55, 55))
    #    else:
    #        x0, y0 = np.unravel_index(np.argmin(y.reshape(55, 55)), (55, 55))
    #    initial_guess = (x0, y0, 3, 0, 2)
    #    try:
    #        popt, pcov = opt.curve_fit(vutil.sugar_gaussian_f, 55, y,
    #                                   p0=initial_guess)
    #        #print popt
    #        paras[i, :5] = popt
    #        pred_y = vutil.sugar_gaussian_f(55, *popt)
    #        paras[i, 5] = np.square(y-pred_y).sum()
    #    except RuntimeError:
    #        print 'Error - curve_fit failed'
    #        paras[i, :] = np.nan
    #np.save(os.path.join(cross_corr_dir, 'curve_fit_paras.npy'), paras)
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

