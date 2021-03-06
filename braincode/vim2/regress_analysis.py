# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy import ndimage
from scipy.misc import imsave
from sklearn import linear_model

from braincode.util import configParser
from braincode.math import parallel_corr2_coef, ridge
from braincode.pipeline import retinotopy
from braincode.pipeline.base import random_cross_modal_corr
from braincode.pipeline.base import random_ridge_regression
from braincode.pipeline.base import pred_cnn_ridge
from braincode.vim2 import util as vutil
from braincode.vim2 import dataio


def check_path(dir_path):
    """Check whether the directory does exist, if not, create it."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, 0755)

def ridge_regression(train_x, train_y, val_x, val_y):
    """fMRI encoding model fitting using ridge regression.
    90% training data used for model tuning, and the remaining 10% data used
    for model selection.
    """
    print 'Feature data temporal z-score'
    m = np.mean(train_x, axis=1, keepdims=True)
    s = np.std(train_x, axis=1, keepdims=True)
    train_x = (train_x - m) / (1e-5 + s)
    train_x = train_x.T
    val_x = (val_x -m) / (1e-5 + s)
    val_x = val_x.T
    # split training dataset into model tunning set and model selection set
    tune_x = train_x[:int(7200*0.9), :]
    sel_x = train_x[int(7200*0.9):, :]
    tune_y = train_y[:int(7200*0.9)]
    sel_y = train_y[int(7200*0.9):]
    # alphas config
    alpha_num = 10
    # model fitting
    test_r2 = np.zeros(alpha_num)
    for a in range(alpha_num):
        alpha_list = np.logspace(-2, 3, alpha_num)
        reg = linear_model.Ridge(alpha=alpha_list[a])
        reg.fit(tune_x, tune_y)
        pred_sel_y = reg.predict(sel_x)
        r2 = 1.0 - np.mean(np.square(sel_y - pred_sel_y)) / np.var(sel_y)
        test_r2[a] = r2
    # select the best model
    print 'Max R^2 on test dataset: %s'%(test_r2.max())
    sel_alpha = alpha_list[test_r2.argmax()]
    reg = linear_model.Ridge(alpha=sel_alpha)
    reg.fit(tune_x, tune_y)
    pred_val_y = reg.predict(val_x)
    val_r2 = 1.0 - np.mean(np.square(val_y - pred_val_y))
    print 'R^2 on validation dataset: %s'%(val_r2)
    paras = np.concatenate((np.array([reg.intercept_]), reg.coef_))
    return paras,sel_alpha, test_r2, val_r2

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
    pos_mtx = np.zeros((73728, 2))
    pos_mtx[:] = np.nan
    for i in range(len(vxl_idx)):
        print 'Iter %s of %s' %(i+1, len(vxl_idx)),
        tmp = corr_mtx[i, :]
        tmp = np.nan_to_num(np.array(tmp))
        # significant threshold for one-tail test
        tmp[tmp <= 0.019257] = 0
        if np.sum(tmp):
            tmp = tmp.reshape(96, 27, 27)
            mmtx = np.max(tmp, axis=0)
            print mmtx.min(), mmtx.max()
            # get indices of n maximum values
            max_n = 20
            row_idx, col_idx = np.unravel_index(
                                        np.argsort(mmtx.ravel())[-1*max_n:],
                                        mmtx.shape)
            nmtx = np.zeros(mmtx.shape)
            nmtx[row_idx, col_idx] = mmtx[row_idx, col_idx]
            if figout:
                fig_file = os.path.join(fig_dir, 'v'+str(vxl_idx[i])+'.png')
                imsave(fig_file, nmtx)
            # center of mass
            x, y = ndimage.measurements.center_of_mass(nmtx)
            pos_mtx[vxl_idx[i], :] = [x, y]
        else:
            print ' '
    #receptive_field_file = os.path.join(data_dir, 'receptive_field_pos.npy')
    #np.save(receptive_field_file, pos_mtx)
    #pos_mtx = np.load(receptive_field_file)
    # eccentricity
    dist = retinotopy.coord2ecc(pos_mtx, (27, 27))
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
        elif dist[i] < 2.7:
            ecc[i] = 1
        elif dist[i] < 5.4:
            ecc[i] = 2
        elif dist[i] < 8.1:
            ecc[i] = 3
        elif dist[i] < 10.8:
            ecc[i] = 4
        else:
            ecc[i] = 5
    #dist_vec = np.nan_to_num(ecc)
    #vol = dist_vec.reshape(18, 64, 64)
    vol = ecc.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max' + str(max_n) + '_ecc.nii.gz'))
    # angle
    angle_vec = retinotopy.coord2angle(pos_mtx, (27, 27))
    #angle_vec = np.nan_to_num(angle_vec)
    vol = angle_vec.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir,
                                'train_max'+ str(max_n) + '_angle.nii.gz'))

def ridge_retinotopic_mapping(corr_file, vxl_idx=None, top_n=None):
    """Make the retinotopic mapping using activation map from CNN."""
    data_dir = os.path.dirname(corr_file)
    # load the cross-correlation matrix from file
    corr_mtx = np.load(corr_file, mmap_mode='r')
    # corr_mtx.shape = (3025, vxl_num)
    if not isinstance(vxl_idx, np.ndarray):
        vxl_idx = np.arange(corr_mtx.shape[1])
    if not top_n:
        top_n = 20
    pos_mtx = np.zeros((73728, 2))
    pos_mtx[:] = np.nan
    for i in range(len(vxl_idx)):
        print 'Iter %s of %s' %(i, len(vxl_idx)),
        tmp = corr_mtx[:, i]
        tmp = np.nan_to_num(np.array(tmp))
        # significant threshold
        # one-tail test
        #tmp[tmp <= 0.17419] = 0
        if np.sum(tmp):
            tmp = tmp.reshape(55, 55)
            print tmp.min(), tmp.max()
            # get indices of n maximum values
            row, col = np.unravel_index(np.argsort(tmp.ravel())[-1*top_n:],
                                        tmp.shape)
            mtx = np.zeros(tmp.shape)
            mtx[row, col] = tmp[row, col]
            # center of mass
            x, y = ndimage.measurements.center_of_mass(mtx)
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
    vol = ecc.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir, 'ecc_max%s.nii.gz'%(top_n)))
    # angle
    angle_vec = retinotopy.coord2angle(pos_mtx, (55, 55))
    vol = angle_vec.reshape(18, 64, 64)
    vutil.save2nifti(vol, os.path.join(data_dir, 'angle_max%s.nii.gz'%(top_n)))

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
    db_dir = os.path.join(cf.get('database', 'path'), 'vim2')
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures', 'vim2', 'caffenet')
    res_dir = os.path.join(root_dir, 'subjects')
 
    # layer info
    layer_info = {'conv1': [96, 55, 55],
                  'norm1': [96, 27, 27],
                  'conv2': [256, 27, 27],
                  'norm2': [256, 13, 13],
                  'conv3': [384, 13, 13],
                  'conv4': [384, 13, 13],
                  'conv5': [256, 13, 13],
                  'pool5': [256, 6, 6]}

    # subj config
    subj_id = 1
    roi = 'v1rh'
    layer = 'conv1'
    layer_size = layer_info[layer]
    subj_dir = os.path.join(res_dir, 'vim2_S%s'%(subj_id))
    roi_dir = os.path.join(subj_dir, 'ridge', roi)
    check_path(roi_dir)

    # load fmri data
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim2_fmri(db_dir, subj_id,
                                                                roi=roi)

    #-- load cnn activation data
    # norm1 data shape = (96, 27, 27, 7200/540)
    train_feat_file = os.path.join(feat_dir, '%s_train_trs.npy'%(layer))
    train_feat_ts = np.load(train_feat_file, mmap_mode='r')
    val_feat_file = os.path.join(feat_dir, '%s_val_trs.npy'%(layer))
    val_feat_ts = np.load(val_feat_file, mmap_mode='r')
 
    #-- Cross-modality mapping: voxel~CNN unit corrlation
    corr_file = os.path.join(roi_dir, 'train_%s_corr.npy'%(layer))
    feat_ts = train_feat_ts.reshape(-1, 7200)
    parallel_corr2_coef(train_fmri_ts, feat_ts, corr_file, block_size=96)
    #-- random cross-modal correlation
    #rand_corr_file = os.path.join(roi_dir, 'rand_train_norm1_corr.npy')
    #feat_ts = tr_mag_ts.reshape(16384, 7200)
    #random_cross_modal_corr(train_fmri_ts, feat_ts, 1000, 1000, rand_corr_file)

    #-- multiple regression
    # reshape train feature vector
    corr_file = os.path.join(roi_dir, 'train_%s_corr.npy'%(layer))
    corr_mtx = np.load(corr_file, mmap_mode='r')
    corr_mtx = corr_mtx.reshape(corr_mtx.shape[0], layer_size[0],
                                layer_size[1], layer_size[2])
    reg_wts = np.zeros((corr_mtx.shape[0], layer_size[0]+1))
    test_r2s = np.zeros((corr_mtx.shape[0], 10))
    val_r2s = np.zeros(corr_mtx.shape[0])
    val_alphas = np.zeros(corr_mtx.shape[0])
    for i in range(corr_mtx.shape[0]):
        corr = np.array(corr_mtx[i], dtype=np.float64)
        mcorr = corr.max(axis=0)
        prf = (mcorr - mcorr.min()) / (mcorr.max() - mcorr.min())
        prf = prf.reshape(1, -1)
        train_x = np.zeros((layer_size[0], 7200))
        val_x = np.zeros((layer_size[0], 540))
        for j in range(layer_size[0]):
            feat_ts = np.array(train_feat_ts[j], dtype=np.float64)
            feat_ts = feat_ts.reshape(layer_size[1]*layer_size[2], 7200)
            train_x[j] = np.dot(prf, feat_ts)
            feat_ts = np.array(val_feat_ts[j], dtype=np.float64)
            feat_ts = feat_ts.reshape(layer_size[1]*layer_size[2], 540)
            val_x[j] = np.dot(prf, feat_ts)
        train_y = train_fmri_ts[i]
        val_y = val_fmri_ts[i]
        wts, sel_alpha, test_r2, val_r2 = ridge_regression(train_x, train_y,
                                                           val_x, val_y)
        reg_wts[i] = wts
        test_r2s[i] = test_r2
        val_r2s[i] = val_r2
        val_alphas[i] = sel_alpha
    np.save(os.path.join(roi_dir, '%s_ridge_wts.npy'%(layer)), reg_wts)
    np.save(os.path.join(roi_dir, '%s_ridge_test_r2.npy'%(layer)), test_r2s)
    np.save(os.path.join(roi_dir, '%s_ridge_val_r2.npy'%(layer)), val_r2s)
    np.save(os.path.join(roi_dir, '%s_ridge_alphas.npy'%(layer)), val_alphas)

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
    #tmp_train_file = os.path.join(feat_dir, 'train_norm1_trs_z.npy')
    #np.save(tmp_train_file, train_feat_ts)
    #del train_feat_ts
    #tmp_val_file = os.path.join(feat_dir, 'val_norm1_trs_z.npy')
    #np.save(tmp_val_file, val_feat_ts)
    #del val_feat_ts
    #train_feat_ts = np.load(tmp_train_file, mmap_mode='r')
    #train_feat_ts = train_feat_ts.reshape(69984, 7200)
    #val_feat_ts = np.load(tmp_val_file, mmap_mode='r')
    #val_feat_ts = val_feat_ts.reshape(69984, 540)

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

    #-- pixel-wise random regression
    #selected_vxl_idx = [5666, 9697, 5533, 5597, 5285, 5538, 5273, 5465, 38695,
    #                    38826, 42711, 46873, 30444, 34474, 38548, 42581, 5097,
    #                    5224, 5205, 9238, 9330, 13169, 17748, 21780]
    #train_fmri_ts = np.nan_to_num(train_fmri_ts[selected_vxl_idx])
    #val_fmri_ts = np.nan_to_num(val_fmri_ts[selected_vxl_idx])
    #print 'fmri data temporal z-score'
    #m = np.mean(train_fmri_ts, axis=1, keepdims=True)
    #s = np.std(train_fmri_ts, axis=1, keepdims=True)
    #train_fmri_ts = (train_fmri_ts - m) / (1e-10 + s)
    #val_fmri_ts = (val_fmri_ts - m) / (1e-10 + s)
    #ridge_prefix = 'random_conv3_pixel_wise'
    #random_ridge_regression(train_feat_ts, train_fmri_ts,
    #                        val_feat_ts, val_fmri_ts,
    #                        1000, ridge_dir, ridge_prefix)
    #-- permutation stats
    #rand_f = os.path.join(ridge_dir,'random_conv3_pixel_wise_corr.npy')
    #random_corr_mtx = np.load(rand_f)
    #permutation_stats(random_corr_mtx)
    
    #-- CNN activation prediction models
    #cnn_pred_dir = os.path.join(subj_dir, 'cnn_pred')
    #check_path(cnn_pred_dir)
    #pred_out_prefix = 'pred_norm1'
    #pred_cnn_ridge(train_fmri_ts, train_feat_ts, val_fmri_ts, val_feat_ts,
    #               cnn_pred_dir, pred_out_prefix, with_wt=True, n_cpus=2)
    #-- cnn features reconstruction
    #wt_file = os.path.join(cnn_pred_dir, pred_out_prefix+'_weights.npy')
    #wts = np.load(wt_file, mmap_mode='r')
    #pred_val_feat_ts_z = wts.dot(val_fmri_ts)
    #print pred_val_feat_ts_z.shape
    #pred_val_feat_ts = pred_val_feat_ts_z*(1e-10+train_feat_s) + train_feat_m
    #out_file = os.path.join(cnn_pred_dir, pred_out_prefix+'_val_feat_ts.npy')
    #np.save(out_file, np.array(pred_val_feat_ts))

    #-- close fmri data
    #tf.close()

