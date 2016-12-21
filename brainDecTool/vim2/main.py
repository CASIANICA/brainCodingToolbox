# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
from scipy import ndimage
from scipy.misc import imsave
from scipy.stats import chisqprob

from brainDecTool.math import rcca
from sklearn.cross_decomposition import PLSCanonical

from brainDecTool.util import configParser
from brainDecTool.math import parallel_corr2_coef, corr2_coef
from brainDecTool.pipeline import retinotopy
from brainDecTool.pipeline.base import random_cross_modal_corr
from brainDecTool.pipeline.base import multiple_regression
from brainDecTool.pipeline.base import ridge_regression
import util as vutil


def roi2nifti(fmri_table, filename, mode='full'):
    """Save ROI as a nifti file.
    `mode`: 'full' for whole ROIs mask creation.
            'small' for mask creation for alignment.
    """
    if mode=='full':
        roi_label = {'v1lh': 1, 'v1rh': 2, 'v2lh': 3, 'v2rh': 4,
                     'v3lh': 5, 'v3rh': 6, 'v3alh': 7, 'v3arh': 8,
                     'v3blh': 9, 'v3brh': 10, 'v4lh': 11, 'v4rh': 12,
                     'latocclh': 13, 'latoccrh': 14, 'VOlh': 15, 'VOrh': 16,
                    'STSlh': 17, 'STSrh': 18, 'RSClh': 19, 'RSCrh': 20,
                    'PPAlh': 21, 'PPArh': 22, 'OBJlh': 23, 'OBJrh': 24,
                    'MTlh': 25, 'MTrh': 26, 'MTplh': 27, 'MTprh': 28,
                    'IPlh': 29, 'IPrh': 30, 'FFAlh': 31, 'FFArh': 32,
                    'EBAlh': 33, 'EBArh': 34, 'OFAlh': 35, 'OFArh': 36,
                    'v7alh': 37, 'v7arh': 38, 'v7blh': 39, 'v7brh': 40,
                    'v7clh': 41, 'v7crh': 42, 'v7lh': 43, 'v7rh': 44,
                    'IPS1lh': 45, 'IPS1rh': 46, 'IPS2lh': 47, 'IPS2rh': 48,
                    'IPS3lh': 49, 'IPS3rh': 50, 'IPS4lh': 51, 'IPS4rh': 52,
                    'MSTlh': 53, 'MSTrh': 54, 'TOSlh': 55, 'TOSrh': 56}
    else:
        roi_label = {'v1lh': 1, 'v1rh': 2, 'v2lh': 3, 'v2rh': 4,
                     'v3lh': 5, 'v3rh': 6, 'v3alh': 7, 'v3arh': 8,
                     'v3blh': 9, 'v3brh': 10, 'v4lh': 11, 'v4rh': 12,
                    'MTlh': 13, 'MTrh': 14, 'MTplh': 15, 'MTprh': 16}

    roi_list = fmri_table.list_nodes('/roi')
    roi_shape = roi_list[0].shape
    roi_mask = np.zeros(roi_shape)
    roi_list = [r.name for r in roi_list if r.name in roi_label]
    for r in roi_list:
        roi_mask += fmri_table.get_node('/roi/%s'%(r))[:] * roi_label[r]
    vutil.save2nifti(roi_mask, filename)

def get_roi_mask(fmri_table, nifti=False):
    """Save ROIs as a mask."""
    roi_list = fmri_table.list_nodes('/roi')
    roi_shape = roi_list[0].shape
    mask = np.zeros(roi_shape)
    for r in roi_list:
        mask += fmri_table.get_node('/roi/%s'%(r.name))[:]
    if nifti:
        vutil.save2nifti(mask, 'all_roi_mask.nii.gz')
    else:
        return mask.flatten()

def gen_mean_vol(fmri_table, dataset, filename):
    """Make a mean response map as a reference volume."""
    data = fmri_table.get_node('/'+dataset)[:]
    # replace nan to zero
    data = np.nan_to_num(data)
    mean_data = np.mean(data, axis=1)
    vol = np.zeros((18, 64, 64))
    
    for i in range(data.shape[0]):
        c = vutil.idx2coord(i)
        vol[c[0], c[1], c[2]] = mean_data[i]
    
    vutil.save2nifti(vol, filename)

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

def plscorr(train_fmri_ts, train_feat_ts, val_fmri_ts, val_feat_ts,
            out_dir, mask_file):
    """Compute PLS correlation between brain activity and CNN activation."""
    train_feat_ts = train_feat_ts.reshape(-1, train_feat_ts.shape[3]).T
    val_feat_ts = val_feat_ts.reshape(-1, val_feat_ts.shape[3]).T
    train_fmri_ts = train_fmri_ts.T
    val_fmri_ts = val_fmri_ts.T

    ## Iteration loop for different component number
    #for n in range(5, 15):
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

    plsca = PLSCanonical(n_components=10)
    plsca.fit(train_feat_ts, train_fmri_ts)
    #mask_file = '/Users/sealhuang/brainDecoding/S1_mask.nii.gz'
    from sklearn.externals import joblib
    joblib.dump(plsca, os.path.join(out_dir, 'plsca_model.pkl'))
    plsca = joblib.load(os.path.join(out_dir, 'plsca_model.pkl'))

    # prediction score
    pred_fmri_ts = plsca.predict(val_feat_ts)
    # calculate correlation coefficient between truth and prediction
    fmri_pred_r = corr2_coef(val_fmri_ts.T, pred_fmri_ts.T, mode='pair')
    mask = vutil.data_swap(mask_file)
    vxl_idx = np.nonzero(mask.flatten()==1)[0]
    tmp = np.zeros_like(mask.flatten(), dtype=np.float64)
    tmp[vxl_idx] = fmri_pred_r
    tmp = tmp.reshape(mask.shape)
    vutil.save2nifti(tmp, os.path.join(out_dir, 'pred_fmri_r.nii.gz'))
    
    # get PLS-CCA weights
    feat_cc, fmri_cc = plsca.transform(train_feat_ts, train_fmri_ts)
    np.save(os.path.join(out_dir, 'feat_cc.npy'), feat_cc)
    np.save(os.path.join(out_dir, 'fmri_cc.npy'), fmri_cc)
    feat_weight = plsca.x_weights_.reshape(96, 11, 11, 10)
    fmri_weight = plsca.y_weights_
    np.save(os.path.join(out_dir, 'feat_weights.npy'), feat_weight)
    np.save(os.path.join(out_dir, 'fmri_weights.npy'), fmri_weight)
    vutil.plot_cca_fweights(feat_weight, out_dir, 'feat_weight', abs_flag=False)
    vutil.save_cca_volweights(fmri_weight, mask_file, out_dir)
    
    # calculate corr between original variables and the CCs
    feat_cc = np.load(os.path.join(out_dir, 'feat_cc.npy'))
    parallel_corr2_coef(train_feat_ts.T, feat_cc.T, 
                        os.path.join(out_dir, 'feat_cc_corr.npy'),
                        block_size=10, n_jobs=1)
    feat_cc_corr = np.load(os.path.join(out_dir, 'feat_cc_corr.npy'))
    feat_cc_corr = feat_cc_corr.reshape(96, 11, 11, 10)
    vutil.plot_cca_fweights(feat_cc_corr, out_dir, 'feat_cc_corr',
                            abs_flag=False)
    #vutil.fweights_bar(feat_cc_corr)
    #fmri_cc = np.load(os.path.join(out_dir, 'fmri_cc.npy'))
    #parallel_corr2_coef(train_fmri_ts.T, fmri_cc.T,
    #                    os.path.join(out_dir, 'fmri_cc_corr.npy'),
    #                    block_size=10, n_jobs=1)
    #fmri_cc_corr = np.load(os.path.join(out_dir, 'fmri_cc_corr.npy'))
    #vutil.save_cca_volweights(fmri_cc_corr, mask_file, out_dir,
    #                          prefix_name='fmri_cc_corr')

    ## Chi-square test -- not suit for small sample size application ... 
    #rlist = []
    #for i in range(components_num):
    #    r = np.corrcoef(feat_c[:, i], fmri_c[:, i])[0, 1]
    #    rlist.append(r)
    #print 'Correlation coefficient', rlist
    #print 'Chi-square test ...'
    #r2list = [1-r**2 for r in rlist]
    #print '1-r^2:', r2list
    #m = feat_ts.shape[1]
    #n = fmri_ts.shape[1]
    #p = feat_ts.shape[0]
    #for i in  range(components_num):
    #    chi2 = ((m+n+3)*1.0/2-p)*np.log(reduce(lambda x, y: x*y, r2list[i:]))
    #    dof = (m-i)*(n-i)
    #    print 'Canonical component %s, p value: %s'%(i+1, chisqprob(chi2, dof))


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
    vutil.plot_cca_fweights(feat_pred_r, out_dir,
                            'pred_feat_r_CC%s'%(CCnum), abs_flag=False)
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
    vutil.plot_cca_fweights(feat_weights, out_dir, 'feat_weight_CC%s'%(CCnum),
                            abs_flag=False)
    vutil.save_cca_volweights(fmri_weights, mask_file, out_dir)

    feat_cc = cca.comps[0]
    parallel_corr2_coef(train_feat_ts.T, feat_cc.T, 
                        os.path.join(out_dir, 'feat_cc_corr.npy'),
                        block_size=7, n_jobs=1)
    feat_cc_corr = np.load(os.path.join(out_dir, 'feat_cc_corr.npy'))
    feat_cc_corr = feat_cc_corr.reshape(96, 11, 11, 7)
    vutil.plot_cca_fweights(feat_cc_corr, out_dir, 'feat_cc_corr',
                            abs_flag=False)

if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures')
    db_dir = os.path.join(root_dir, 'subjects')

    # subj config
    subj_id = 3
    subj_dir = os.path.join(db_dir, 'vS%s'%(subj_id))
    
    #-- load fmri data
    fmri_file = os.path.join(subj_dir, 'VoxelResponses_subject%s.mat'%(subj_id))
    tf = tables.open_file(fmri_file)
    #tf.list_nodes
    #-- roi mat to nii
    #roi_file = os.path.join(subj_dir, 'S%s_small_roi.nii.gz'%(subj_id))
    #roi2nifti(tf, roi_file, mode='small')
    #-- get mean fmri responses
    #dataset = 'rt'
    #mean_file = os.path.join(subj_dir, 'S%s_mean_%s.nii.gz'%(subj_id, dataset))
    #gen_mean_vol(tf, dataset, mean_file)

    #-- calculate cross-modality corrlation 
    # load fmri response from training/validation dataset
    train_fmri_ts = tf.get_node('/rt')[:]
    val_fmri_ts = tf.get_node('/rv')[:]
    # data.shape = (73728, 540/7200)
    # load brain mask
    mask_file = os.path.join(subj_dir, 'S%s_mask.nii.gz'%(subj_id))
    mask = vutil.data_swap(mask_file).flatten()
    vxl_idx = np.nonzero(mask==1)[0]
    train_fmri_ts = np.nan_to_num(train_fmri_ts[vxl_idx])
    val_fmri_ts = np.nan_to_num(val_fmri_ts[vxl_idx])
    # load convolved cnn activation data for validation dataset
    train_feat1_file = os.path.join(feat_dir, 'feat1_train_trs_ds5.npy')
    train_feat1_ts = np.load(train_feat1_file, mmap_mode='r')
    val_feat1_file = os.path.join(feat_dir, 'feat1_val_trs_ds5.npy')
    val_feat1_ts = np.load(val_feat1_file, mmap_mode='r')
    # data.shape = (96, 55, 55, 540/7200)
    
    #-- retinotopic mapping
    # sum up all channels
    # select parts of channels
    #feat1_ts = feat1_ts[0:48, :]
    #feat1_ts = feat1_ts.sum(axis=0)
    #retino_dir = os.path.join(subj_dir, 'retinotopic')
    #if not os.path.exists(retino_dir):
    #    os.mkdir(retino_dir, 0755)
    #corr_file = os.path.join(retino_dir, 'val_feat1_corr.npy')
    #val_feat1_ts = val_feat1_ts.reshape(290400, 540)
    #parallel_corr2_coef(val_fmri_ts, val_feat1_ts, corr_file, block_size=96)
    #rand_corr_file = os.path.join(retino_dir, 'train_fmri_feat1_rand_corr.npy')
    #random_cross_modal_corr(fmri_ts, feat1_ts, 10, 1000, rand_corr_file)
    #retinotopic_mapping(corr_file, mask)
    
    #-- multiple regression voxel ~ channels from each location
    #regress_file = os.path.join(retino_dir, 'val_fmri_feat1_regress.npy')
    #roi_mask = get_roi_mask(tf)
    #multiple_regression(fmri_ts, feat1_ts, regress_file)

    #-- ridge regression
    #ridge_regression(train_feat, train_fmri, val_feat, val_fmri, outfile)

    #-- PLS-CCA
    pls_dir = os.path.join(subj_dir, 'plscca')
    if not os.path.exists(pls_dir):
        os.mkdir(retino_dir, 0755)
    cca_dir = os.path.join(pls_dir, 'an_layer1')
    if not os.path.exists(cca_dir):
        os.mkdir(cca_dir, 0755)
    plscorr(train_fmri_ts, train_feat1_ts, val_fmri_ts, val_feat1_ts,
            cca_dir, mask_file)

    #-- regularized CCA
    #cca_dir = os.path.join(retino_dir, 'rcca', 'rcca_cc7')
    #if not os.path.exists(cca_dir):
    #    os.mkdir(cca_dir, 0755)
    #reg_cca(train_fmri_ts, train_feat1_ts, val_fmri_ts, val_feat1_ts, cca_dir)

    #-- close fmri data
    tf.close()

