# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
import bob.ip.gabor
from joblib import Parallel, delayed
from sklearn import linear_model

from braincode.util import configParser
from braincode.math import make_2d_gaussian, ridge
from braincode.math.norm import zscore
from braincode.prf import dataio
from braincode.prf import util as vutil
from braincode.pipeline import retinotopy


def check_path(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, 0755)

def get_gabor_features(img):
    """Get Gabor features from input image."""
    img = img.astype(np.float64)
    gwt = bob.ip.gabor.Transform(number_of_scales=9)
    trafo_img = gwt(img)
    out_feat = np.zeros((72, 500, 500))
    for i in range(trafo_img.shape[0]):
        real_p = np.real(trafo_img[i, ...])
        imag_p = np.imag(trafo_img[i, ...])
        out_feat[i, ...] = np.sqrt(np.square(real_p)+np.square(imag_p))
    return out_feat

def get_stim_features(db_dir, feat_dir, data_type):
    """Stimuli processing."""
    sti_dir = os.path.join(db_dir, 'stimuli')
    prefix = {'train': 'Stimuli_Trn_FullRes', 'val': 'Stimuli_Val_FullRes'}
    if data_type == 'train':
        for i in range(15):
            mat_file = os.path.join(sti_dir, prefix['train']+'_%02d.mat'%(i+1))
            print 'Load file %s ...'%(mat_file)
            tf = tables.open_file(mat_file)
            imgs = tf.get_node('/stimTrn')[:]
            tf.close()
            # output matrix: image-number x channel x row x col
            print 'image size %s'%(imgs.shape[2])
            out_features = np.zeros((imgs.shape[2], 72, 500, 500))
            for j in range(imgs.shape[2]):
                x = imgs[..., j].T
                out_features[j, ...] = get_gabor_features(x)
            out_file = prefix['train']+'_%02d_gabor_features.npy'%(i+1)
            out_file = os.path.join(feat_dir, out_file)
            np.save(out_file, out_features)
    else:
        mat_file = os.path.join(sti_dir, prefix['val']+'.mat')
        print 'Load file %s ...'%(mat_file)
        tf = tables.open_file(mat_file)
        imgs = tf.get_node('/stimVal')[:]
        # output matrix: image-number x channel x row x col
        out_features = np.zeros((imgs.shape[2], 72, 500, 500))
        for j in range(imgs.shape[2]):
            x = imgs[..., j].T
            out_features[j, ...] = get_gabor_features(x)
        out_file = prefix['val']+'_gabor_features.npy'
        out_file = os.path.join(feat_dir, out_file)
        np.save(out_file, out_features)

def get_candidate_model(feat_dir, data_type):
    """Get gaussian kernel based on receptivefield features."""
    # derived gauusian-kernel based features
    # candidate pooling centers are spaces 0.4 degrees apart (10 pixels)
    # candidate pooling fields included 17 radii (1, 5, 10, 15, 20, ..., 
    # 55, 60, 70, 80, 90, 100 pixels) between 0.04 degree (1 pixel) and 4
    # degree (100 pixels)
    prefix = {'train': 'Stimuli_Trn_FullRes', 'val': 'Stimuli_Val_FullRes'}
    if data_type=='train':
        time_count = 0
        for i in range(15):
            tmp_file = os.path.join(feat_dir,
                    prefix['train']+'_%02d_gabor_features.npy'%(i+1))
            tmp = np.load(tmp_file, mmap_mode='r')
            time_count += tmp.shape[0]
            out_file = os.path.join(feat_dir,
                                    'train_candidate_model_%02d.npy'%(i+1))
            cand_model = np.memmap(out_file, dtype='float16', mode='w+',
                    shape=(50*50*17, tmp.shape[0], 72))
            Parallel(n_jobs=4)(delayed(model_pro)(tmp, cand_model, xi, yi, si)
                    for si in range(17) for xi in range(50) for yi in range(50))
            # save memmap object as a numpy.array
            model_array = np.array(cand_model)
            np.save(out_file, model_array)
        # merge parts
        print 'Time series length: %s'%(time_count)
        out_file = os.path.join(feat_dir, 'train_candidate_model.npy')
        cand_model = np.memmap(out_file, dtype='float16', mode='w+',
                               shape=(50*50*17, time_count, 72))
        c = 0
        for i in range(15):
            pf = os.path.join(feat_dir, 'train_candidate_model_%02d.npy'%(i+1))
            data = np.load(pf)
            cand_model[:, c:(c+data.shape[1]), :] = data
            c += data.shape[1]
        # save memmap object as a numpy.array
        model_array = np.array(cand_model)
        np.save(out_file, model_array)
    else:
        tmp_file = os.path.join(feat_dir, prefix['val']+'_gabor_features.npy')
        tmp = np.load(tmp_file, mmap_mode='r')
        time_count = tmp.shape[0]
        out_file = os.path.join(feat_dir, '%s_candidate_model.npy'%(data_type))
        cand_model = np.memmap(out_file, dtype='float16', mode='w+',
                               shape=(50*50*17, time_count, 72))
        Parallel(n_jobs=4)(delayed(model_pro)(tmp, cand_model, xi, yi, si)
                    for si in range(17) for xi in range(50) for yi in range(50))
        # save memmap object as a numpy.array
        model_array = np.array(cand_model)
        np.save(out_file, model_array)

def model_pro(feat, cand_model, xi, yi, si):
    """Sugar function for generating candidate model."""
    mi = si*50*50 + xi*50 + yi
    center_x = np.arange(5, 500, 10)
    center_y = np.arange(5, 500, 10)
    sigma = [1] + [n*5 for n in range(1, 13)] + [70, 80, 90, 100]
    x0 = center_x[xi]
    y0 = center_y[yi]
    s = sigma[si]
    print 'Model %s : center - (%s, %s), sigma %s'%(mi, y0, x0, s)
    kernel = make_2d_gaussian(500, s, center=(x0, y0))
    kernel = kernel.flatten()
    idx_head = 0
    parts = feat.shape[0] / 10
    for i in range(parts):
        tmp = feat[(i*10):(i*10+10), ...]
        tmp = tmp.reshape(720, 250000)
        res = tmp.dot(kernel).astype(np.float16)
        cand_model[mi, idx_head:(idx_head+10), ...] = res.reshape(10, 72)
        idx_head += 10

def get_candidate_model_new(db_dir, data_type):
    """Get gaussian kernel based on receptivefield features."""
    # derived gauusian-kernel based features
    # candidate pooling centers are spaces 0.4 degrees apart (5 pixels)
    # candidate pooling fields included 10 radii (2, 4, 8, 16, 32, 60,
    # 70, 80, 90, 100 pixels) between 0.16 degree (0.08 per pixel) and 8
    # degree (100 pixels)
    img_num = {'train': 1750, 'val': 120}
    feat_file = os.path.join(db_dir, data_type+'_gabor_feat.memdat')
    feat = np.memmap(feat_file, dtype='float32', mode='r',
                     shape=(img_num[data_type], 250, 250, 72))
    out_file = os.path.join(db_dir, '%s_candidate_model.npy'%(data_type))
    cand_model = np.memmap(out_file, dtype='float16', mode='w+',
                           shape=(50*50*10, img_num[data_type], 72))
    Parallel(n_jobs=4)(delayed(model_pro_new)(feat, cand_model, xi, yi, si)
                for si in range(10) for xi in range(50) for yi in range(50))
    # save memmap object as a numpy.array
    model_array = np.array(cand_model)
    np.save(out_file, model_array)

def model_pro_new(feat, cand_model, xi, yi, si):
    """Sugar function for generating candidate model."""
    mi = si*50*50 + xi*50 + yi
    center_x = np.arange(2, 250, 5)
    center_y = np.arange(2, 250, 5)
    sigma = [2, 4, 8, 16, 32, 60, 70, 80, 90, 100]
    x0 = center_x[xi]
    y0 = center_y[yi]
    s = sigma[si]
    print 'Model %s : center - (%s, %s), sigma %s'%(mi, y0, x0, s)
    kernel = make_2d_gaussian(250, s, center=(x0, y0))
    kernel = kernel.flatten()
    idx_head = 0
    parts = feat.shape[0] / 10
    for i in range(parts):
        tmp = feat[(i*10):(i*10+10), ...]
        tmp = np.transpose(tmp, (0, 3, 1, 2))
        tmp = tmp.reshape(720, 62500)
        res = tmp.dot(kernel).astype(np.float16)
        cand_model[mi, idx_head:(idx_head+10), ...] = res.reshape(10, 72)
        idx_head += 10

def get_vxl_idx(prf_dir, db_dir, subj_id, roi):
    """Get voxel index in specific ROI"""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    print 'Voxel number: %s'%(len(vxl_idx))
    roi_dir = os.path.join(prf_dir, roi)
    check_path(roi_dir)
    np.save(os.path.join(roi_dir, 'vxl_idx.npy'), vxl_idx)

def ridge_fitting(feat_dir, prf_dir, db_dir, subj_id, roi):
    """pRF model fitting using ridge regression.
    90% trainning data used for model tuning, and another 10% data used for
    model seletion.
    """
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del val_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    train_models = np.load(os.path.join(feat_dir, 'train_candidate_model.npy'),
                           mmap_mode='r')
    # output directory config
    roi_dir = os.path.join(prf_dir,  roi)
    check_path(roi_dir)
    # model seletion and tuning
    ALPHA_NUM = 20
    BOOTS_NUM = 15
    paras_file = os.path.join(roi_dir, 'reg_paras.npy')
    paras = np.memmap(paras_file, dtype='float64', mode='w+',
                      shape=(42500, len(vxl_idx), 72))
    mcorr_file= os.path.join(roi_dir, 'reg_model_corr.npy')
    mcorr = np.memmap(mcorr_file, dtype='float64', mode='w+',
                      shape=(42500, len(vxl_idx)))
    alphas_file = os.path.join(roi_dir, 'reg_alphas.npy')
    alphas = np.memmap(alphas_file, dtype='float64', mode='w+',
                       shape=(42500, len(vxl_idx)))
    # fMRI data z-score
    print 'fmri data temporal z-score'
    m = np.mean(train_fmri_ts, axis=1, keepdims=True)
    s = np.std(train_fmri_ts, axis=1, keepdims=True)
    train_fmri_ts = (train_fmri_ts - m) / (1e-10 + s)
    # split training dataset into model tunning set and model selection set
    tune_fmri_ts = train_fmri_ts[:, :int(1750*0.9)]
    sel_fmri_ts = train_fmri_ts[:, int(1750*0.9):]
    # model testing
    for i in range(42500):
        print 'Model %s'%(i)
        # remove models which centered outside the 20 degree of visual angle
        xi = (i % 2500) / 50
        yi = (i % 2500) % 50
        x0 = np.arange(5, 500, 10)[xi]
        y0 = np.arange(5, 500, 10)[yi]
        d = np.sqrt(np.square(x0-250)+np.square(y0-250))
        if d > 249:
            print 'Model center outside the visual angle'
            paras[i, ...] = np.NaN
            mcorr[i] = np.NaN
            alphas[i] = np.NaN
            continue
        train_x = np.array(train_models[i, ...]).astype(np.float64)
        train_x = zscore(train_x.T).T
        # split training dataset into model tunning and selection sets
        tune_x = train_x[:int(1750*0.9), :]
        sel_x = train_x[int(1750*0.9):, :]
        wt, r, alpha, bscores, valinds = ridge.bootstrap_ridge(
                tune_x, tune_fmri_ts.T, sel_x, sel_fmri_ts.T,
                alphas=np.logspace(-2, 3, ALPHA_NUM),
                nboots=BOOTS_NUM, chunklen=175, nchunks=1,
                single_alpha=False, use_corr=False)
        paras[i, ...] = wt.T
        mcorr[i] = r
        alphas[i] = alpha
    # save output
    paras = np.array(paras)
    np.save(paras_file, paras)
    mcorr = np.array(mcorr)
    np.save(mcorr_file, mcorr)
    alphas = np.array(alphas)
    np.save(alphas_file, alphas)

def ridge_regression(prf_dir, db_dir, subj_id, roi):
    """pRF model fitting using ridge regression.
    90% trainning data used for model tuning, and another 10% data used for
    model seletion.
    """
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del val_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    train_models = np.load(os.path.join(db_dir, 'train_candidate_model.npy'),
                           mmap_mode='r')
    # output directory config
    roi_dir = os.path.join(prf_dir,  roi)
    check_path(roi_dir)

    # model seletion and tuning
    ALPHA_NUM = 10
    paras_file = os.path.join(roi_dir, 'reg_paras.npy')
    paras = np.memmap(paras_file, dtype='float64', mode='w+',
                      shape=(len(vxl_idx), 73))
    val_r2_file= os.path.join(roi_dir, 'reg_val_r2.npy')
    val_r2 = np.memmap(val_r2_file, dtype='float64', mode='w+',
                       shape=(len(vxl_idx), 25000, ALPHA_NUM))
    alphas_file = os.path.join(roi_dir, 'reg_alphas.npy')
    alphas = np.memmap(alphas_file, dtype='float64', mode='w+',
                       shape=(len(vxl_idx)))
    # fMRI data z-score
    print 'fmri data temporal z-score'
    m = np.mean(train_fmri_ts, axis=1, keepdims=True)
    s = np.std(train_fmri_ts, axis=1, keepdims=True)
    train_fmri_ts = (train_fmri_ts - m) / (1e-5 + s)
    # split training dataset into model tunning set and model selection set
    tune_fmri_ts = train_fmri_ts[:, :int(1750*0.9)]
    sel_fmri_ts = train_fmri_ts[:, int(1750*0.9):]
    # model fitting
    for i in range(len(vxl_idx)):
        print '-----------------'
        print 'Voxel %s'%(i)
        for j in range(25000):
            #print 'Model %s'%(j)
            # remove models which centered outside the 20 degree of visual angle
            xi = (j % 2500) / 50
            yi = (j % 2500) % 50
            x0 = np.arange(2, 250, 5)[xi]
            y0 = np.arange(2, 250, 5)[yi]
            d = np.sqrt(np.square(x0-125)+np.square(y0-125))
            if d > 124:
                #print 'Model center outside the visual angle'
                paras[i, ...] = np.NaN
                val_r2[i, j, :] = np.NaN
                continue
            train_x = np.array(train_models[j, ...]).astype(np.float64)
            # split training dataset into model tunning and selection sets
            tune_x = train_x[:int(1750*0.9), :]
            sel_x = train_x[int(1750*0.9):, :]
            for a in range(ALPHA_NUM):
                alpha_list = np.logspace(-2, 3, ALPHA_NUM)
                # model fitting
                reg = linear_model.Ridge(alpha=alpha_list[a])
                reg.fit(tune_x, tune_fmri_ts[i])
                val_pred = reg.predict(sel_x)
                ss_tol = np.var(sel_fmri_ts[i]) * 175
                r2 = 1.0 - np.sum(np.square(sel_fmri_ts[i] - val_pred))/ss_tol
                val_r2[i, j, a] = r2
        # select best model
        vxl_r2 = np.nan_to_num(val_r2[i, ...])
        sel_mdl_i, sel_alpha_i = np.unravel_index(vxl_r2.argmax(), vxl_r2.shape)
        train_x = np.array(train_models[sel_mdl_i, ...]).astype(np.float64)
        # split training dataset into model tunning and selection sets
        tune_x = train_x[:int(1750*0.9), :]
        sel_x = train_x[int(1750*0.9):, :]
        alpha_list = np.logspace(-2, 3, ALPHA_NUM)
        # selected model fitting
        reg = linear_model.Ridge(alpha=alpha_list[sel_alpha_i])
        reg.fit(tune_x, tune_fmri_ts[i])
        val_pred = reg.predict(sel_x)
        ss_tol = np.var(sel_fmri_ts[i]) * 175
        r2 = 1.0 - np.sum(np.square(sel_fmri_ts[i] - val_pred))/ss_tol
        print 'r-square recal: %s'%(r2)
        #print 'r-square cal: %s'%(vxl_r2.max())
        paras[i, ...] = np.concatenate((np.array([reg.intercept_]), reg.coef_))
        alphas[i] = alpha_list[sel_alpha_i]
    # save output
    paras = np.array(paras)
    np.save(paras_file, paras)
    val_r2 = np.array(val_r2)
    np.save(val_r2_file, val_r2)
    alphas = np.array(alphas)
    np.save(alphas_file, alphas)

def ridge_regression_model_test(prf_dir, db_dir, subj_id, roi):
    """Test pRF model derived from ridge regression with test dataset."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    val_models = np.load(os.path.join(db_dir, 'val_candidate_model.npy'),
                         mmap_mode='r')
    # fMRI data z-score
    print 'fmri data temporal z-score'
    m = np.mean(val_fmri_ts, axis=1, keepdims=True)
    s = np.std(val_fmri_ts, axis=1, keepdims=True)
    val_fmri_ts = (val_fmri_ts - m) / (1e-5 + s)
    # output directory config
    roi_dir = os.path.join(prf_dir,  roi)
    check_path(roi_dir)

    # load selected models and the corresponding parameters
    val_r2_file = os.path.join(roi_dir, 'reg_val_r2.npy')
    val_r2 = np.load(val_r2_file, mmap_mode='r')
    paras_file = os.path.join(roi_dir, 'reg_paras.npy')
    paras = np.load(paras_file)
    alphas_file = os.path.join(roi_dir, 'reg_alphas.npy')
    alphas = np.load(alphas_file)

    # output var
    test_r2 = np.zeros(len(vxl_idx))
    prf_pos = np.zeros((len(vxl_idx), 3))
    # parameter candidates
    alpha_list = np.logspace(-2, 3, 10)
    sigma = [2, 4, 8, 16, 32, 60, 70, 80, 90, 100]
    for i in range(len(vxl_idx)):
        print '----------------'
        print 'Voxel %s'%(i)
        vxl_r2 = np.nan_to_num(val_r2[i, ...])
        sel_mdl_i, sel_alpha_i = np.unravel_index(vxl_r2.argmax(), vxl_r2.shape)
        print 'Select model %s'%(sel_mdl_i)
        print 'Select alpha value %s - %s'%(alpha_list[sel_alpha_i], alphas[i])
        # get model position info
        xi = (sel_mdl_i % 2500) / 50
        yi = (sel_mdl_i % 2500) % 50
        x0 = np.arange(2, 250, 5)[xi]
        y0 = np.arange(2, 250, 5)[yi]
        s = sigma[sel_mdl_i / 2500]
        prf_pos[i] = np.array([y0, x0, s])
        # compute r^2 using test dataset
        test_x = np.array(val_models[sel_mdl_i, ...]).astype(np.float64)
        test_x = np.concatenate((np.ones((120, 1)), test_x), axis=1)
        val_pred = test_x.dot(paras[i])
        ss_tol = np.var(val_fmri_ts[i]) * 120
        r2 = 1.0 - np.sum(np.square(val_fmri_ts[i] - val_pred))/ss_tol
        print 'r-square on test dataset: %s'%(r2)
        test_r2[i] = r2
    # save output
    np.save(os.path.join(roi_dir, 'reg_prf_test_r2.npy'), test_r2)
    np.save(os.path.join(roi_dir, 'sel_reg_prf_pos.npy'), prf_pos)

def prf_selection(feat_dir, prf_dir, db_dir, subj_id, roi):
    """Select best model for each voxel and validating."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    val_models = np.load(os.path.join(feat_dir, 'val_candidate_model.npy'),
                         mmap_mode='r')
    # output directory config
    roi_dir = os.path.join(prf_dir, roi)
    # load candidate model parameters 
    paras = np.load(os.path.join(roi_dir, 'reg_paras.npy'))
    mcorr = np.load(os.path.join(roi_dir, 'reg_model_corr.npy'))
    alphas = np.load(os.path.join(roi_dir, 'reg_alphas.npy'))
    sel_paras = np.zeros((mcorr.shape[1], 72))
    sel_model = np.zeros(mcorr.shape[1])
    sel_model_corr = np.zeros(mcorr.shape[1])
    for i in range(mcorr.shape[1]):
        maxi = np.argmax(np.nan_to_num(mcorr[:, i]))
        print 'Voxel %s - Max corr %s - Model %s'%(i, mcorr[maxi, i], maxi)
        print 'Alpha : %s'%(alphas[maxi, i])
        sel_paras[i] = paras[maxi, i]
        sel_model[i] = maxi
        feats = np.array(val_models[maxi, ...]).astype(np.float64)
        feats = zscore(feats.T).T
        pred = np.dot(feats, sel_paras[i])
        sel_model_corr[i] = np.corrcoef(pred, val_fmri_ts[i])[0, 1]
        print 'Val Corr : %s'%(sel_model_corr[i])
    np.save(os.path.join(roi_dir, 'reg_sel_paras.npy'), sel_paras)
    np.save(os.path.join(roi_dir, 'reg_sel_model.npy'), sel_model)
    np.save(os.path.join(roi_dir, 'reg_sel_model_corr.npy'), sel_model_corr)

def null_distribution_prf_tunning(feat_dir, prf_dir, db_dir, subj_id, roi):
    """Generate Null distribution of pRF model tunning using validation data."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    val_models = np.load(os.path.join(feat_dir, 'val_candidate_model.npy'),
                         mmap_mode='r')
    # output directory config
    roi_dir = os.path.join(prf_dir, roi)
    # load selected model parameters
    paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    null_corr = np.zeros((paras.shape[0], 1000))
    for i in range(paras.shape[0]):
        print 'Voxel %s'%(i)
        # load features
        feats = np.array(val_models[int(sel_model[i]), ...]).astype(np.float64)
        feats = zscore(feats.T).T
        pred = np.dot(feats, paras[i])
        for j in range(1000):
            shuffled_val_ts = np.random.permutation(val_fmri_ts[i])
            null_corr[i, j] = np.corrcoef(pred, shuffled_val_ts)[0, 1]
    np.save(os.path.join(roi_dir, 'random_corr.npy'), null_corr)

def gabor_contribution2prf(feat_dir, prf_dir, db_dir, subj_id, roi):
    """Calculate tunning contribution of each gabor sub-banks."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load candidate models
    val_models = np.load(os.path.join(feat_dir, 'val_candidate_model.npy'),
                         mmap_mode='r')
    # load selected model parameters
    roi_dir = os.path.join(prf_dir, roi)
    paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    gabor_corr = np.zeros((paras.shape[0], 9))
    for i in range(paras.shape[0]):
        print 'Voxel %s'%(i)
        # load features
        feats = np.array(val_models[int(sel_model[i]), ...]).astype(np.float64)
        feats = zscore(feats.T).T
        for j in range(9):
            pred = np.dot(feats[:, (j*8):(j*8+8)], paras[i, (j*8):(j*8+8)])
            gabor_corr[i, j] = np.corrcoef(pred, val_fmri_ts[i])[0, 1]
    np.save(os.path.join(roi_dir, 'gabor_contributes.npy'), gabor_corr)

def orient_selectivity(prf_dir, roi):
    """Calculate orientation selectivity index for each voxel."""
    # load selected model parameters
    roi_dir = os.path.join(prf_dir, roi)
    paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    osi = np.zeros(paras.shape[0])
    for i in range(paras.shape[0]):
        print 'Voxel %s'%(i)
        # load features
        sel_paras = paras[i].reshape(9, 8)
        orient = sel_paras.sum(axis=0)
        osi[i] = orient.max() - (orient.sum() - orient.max())/7
    np.save(os.path.join(roi_dir, 'orient_selectivity.npy'), osi)

def prf_recon(prf_dir, db_dir, subj_id, roi):
    """Reconstruct pRF based on selected model."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    del val_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # output directory config
    roi_dir = os.path.join(prf_dir, roi)
    # pRF estimate
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    prfs = np.zeros((sel_models.shape[0], 500, 500))
    fig_dir = os.path.join(roi_dir, 'figs')
    check_path(fig_dir)
    for i in range(sel_models.shape[0]):
        # get pRF
        print 'Voxel %s, Val Corr %s'%(i, sel_model_corr[i])
        model_idx = int(sel_models[i])
        # get gaussian pooling field parameters
        si = model_idx / 2500
        xi = (model_idx % 2500) / 50
        yi = (model_idx % 2500) % 50
        x0 = np.arange(5, 500, 10)[xi]
        y0 = np.arange(5, 500, 10)[yi]
        sigma = [1] + [n*5 for n in range(1, 13)] + [70, 80, 90, 100]
        s = sigma[si]
        kernel = make_2d_gaussian(500, s, center=(x0, y0))
        kpos = np.nonzero(kernel)
        paras = sel_paras[i]
        for f in range(9):
            fwt = np.sum(paras[(f*8):(f*8+8)])
            fs = np.sqrt(2)**f*4
            for p in range(kpos[0].shape[0]):
                tmp = make_2d_gaussian(500, fs, center=(kpos[1][p],
                                                        kpos[0][p]))
                prfs[i] += fwt * kernel[kpos[0][p], kpos[1][p]] * tmp
        if sel_model_corr[i]>=0.24:
            prf_file = os.path.join(fig_dir,'Voxel_%s_%s.png'%(i+1, vxl_idx[i]))
            vutil.save_imshow(prfs[i], prf_file)
    np.save(os.path.join(roi_dir, 'prfs.npy'), prfs)

def filter_recon(prf_dir, db_dir, subj_id, roi):
    """Reconstruct filter map of each voxel based on selected model."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    del val_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # output config
    roi_dir = os.path.join(prf_dir, roi)
    # pRF estimate
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    filters = np.zeros((sel_models.shape[0], 500, 500))
    fig_dir = os.path.join(roi_dir, 'filters')
    check_path(fig_dir)
    thr = 0.24
    # gabor bank generation
    gwt = bob.ip.gabor.Transform(number_of_scales=9)
    gwt.generate_wavelets(500, 500)
    spatial_gabors = np.zeros((72, 500, 500))
    for i in range(72):
        w = bob.ip.gabor.Wavelet(resolution=(500, 500),
                        frequency=gwt.wavelet_frequencies[i])
        sw = bob.sp.ifft(w.wavelet.astype(np.complex128)) 
        spatial_gabors[i, ...] = np.roll(np.roll(np.real(sw), 250, 0), 250, 1)
    for i in range(sel_models.shape[0]):
        if sel_model_corr[i]<thr:
            continue
        print 'Voxel %s, Val Corr %s'%(i, sel_model_corr[i])
        model_idx = int(sel_models[i])
        # get gaussian pooling field parameters
        si = model_idx / 2500
        xi = (model_idx % 2500) / 50
        yi = (model_idx % 2500) % 50
        x0 = np.arange(5, 500, 10)[xi]
        y0 = np.arange(5, 500, 10)[yi]
        sigma = [1] + [n*5 for n in range(1, 13)] + [70, 80, 90, 100]
        s = sigma[si]
        print 'center: %s, %s, sigma: %s'%(y0, x0, s)
        kernel = make_2d_gaussian(500, s, center=(x0, y0))
        kpos = np.nonzero(kernel>0.00000001)
        paras = sel_paras[i]
        tmp_file = os.path.join(fig_dir, 'tmp_kernel.npy')
        tmp_filter = np.memmap(tmp_file, dtype='float64', mode='w+',
                               shape=(72, 500, 500))
        Parallel(n_jobs=25)(delayed(filter_pro)(tmp_filter, paras, kernel,
                                                kpos, spatial_gabors, gwt_idx)
                                    for gwt_idx in range(72))
        tmp_filter = np.array(tmp_filter)
        filters[i] = tmp_filter.sum(axis=0)
        os.system('rm %s'%(tmp_file))
        im_file = os.path.join(fig_dir, 'Voxel_%s_%s.png'%(i+1, vxl_idx[i]))
        vutil.save_imshow(filters[i], im_file)
    np.save(os.path.join(roi_dir, 'filters.npy'), filters)

def filter_pro(tmp_filter, paras, kernel, kpos, spatial_gabors, gwt_idx):
    data = np.zeros((500, 500))
    wt = paras[gwt_idx]
    arsw = spatial_gabors[gwt_idx]
    for p in range(kpos[0].shape[0]):
        tmp = img_offset(arsw, (kpos[0][p], kpos[1][p]))
        data += wt * kernel[kpos[0][p], kpos[1][p]] * tmp
    tmp_filter[gwt_idx] = data

def stimuli_recon(prf_dir, db_dir, subj_id, roi):
    """Reconstruct stimulus based on pRF model."""
    # load fmri response
    vxl_idx, train_fmri_ts, val_fmri_ts = dataio.load_vim1_fmri(db_dir, subj_id,
                                                                roi=roi)
    del train_fmri_ts
    print 'Voxel number: %s'%(len(vxl_idx))
    # load model parameters
    roi_dir = os.path.join(prf_dir, roi)
    val_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    filters = np.load(os.path.join(roi_dir, 'filters.npy'))
    recon_imgs = np.zeros((val_fmri_ts.shape[1], 500, 500))
    # fMRI data z-score
    print 'fmri data temporal z-score'
    m = np.mean(val_fmri_ts, axis=1, keepdims=True)
    s = np.std(val_fmri_ts, axis=1, keepdims=True)
    val_fmri_ts = (val_fmri_ts - m) / (1e-10 + s)
    # select significant predicted voxels
    sel_vxls = np.nonzero(val_corr>=0.24)[0]
    for i in range(val_fmri_ts.shape[1]):
        print 'Reconstruct stimilus %s'%(i+1)
        tmp = np.zeros((500, 500))
        for j in sel_vxls:
            tmp += val_fmri_ts[int(j), int(i)] * filters[j]
        recon_imgs[i] = tmp
    np.save(os.path.join(roi_dir, 'recon_img.npy'), recon_imgs)

def retinotopic_mapping(prf_dir, roi):
    """Get eccentricity and angle based on pRF center for each voxel."""
    roi_dir = os.path.join(prf_dir, roi)
    # load selected model index
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    # output variables
    ecc = np.zeros_like(sel_models)
    angle = np.zeros_like(sel_models)
    coords = np.zeros((sel_models.shape[0], 2))
    for i in range(sel_models.shape[0]):
        print 'Voxel %s'%(i+1)
        model_idx = int(sel_models[i])
        xi = (model_idx % 2500) / 50
        yi = (model_idx % 2500) % 50
        # col
        x0 = np.arange(5, 500, 10)[xi]
        # row
        y0 = np.arange(5, 500, 10)[yi]
        coords[i] = [y0, x0]
    # get ecc and angle
    ecc = retinotopy.coord2ecc(coords, 500, 20)
    angle = retinotopy.coord2angle(coords, 500)
    np.save(os.path.join(roi_dir, 'ecc.npy'), ecc)
    np.save(os.path.join(roi_dir, 'angle.npy'), angle)

def img_offset(orig_img, new_center):
    """Move original image to new position based on new center coordinate.
    new_center is (x0, y0), x0 indicates row coordinate, y0 indicates col
    coordinate.
    """
    img_r, img_c = orig_img.shape
    new_img = np.zeros_like(orig_img)
    # move image position based on new center coordinates
    old_x0 = img_r // 2
    old_y0 = img_c // 2
    offset0 = int(np.rint(new_center[0] - old_x0))
    offset1 = int(np.rint(new_center[1] - old_y0))
    pixs = np.mgrid[0:img_r, 0:img_c].reshape(2, img_r*img_c)
    new_x = pixs[0] + offset0
    new_y = pixs[1] + offset1
    pix_idx = (new_x>=0) * (new_x<img_r) * (new_y>=0) * (new_y<img_c)
    new_img[new_x[pix_idx], new_y[pix_idx]] = orig_img[pixs[0, pix_idx],
                                                       pixs[1, pix_idx]]
    return new_img


if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    # database directory config
    #db_dir = os.path.join(cf.get('database', 'path'), 'vim1')
    # directory config for analysis
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures', 'vim1')
    res_dir = os.path.join(root_dir, 'subjects')
    db_dir = os.path.join(root_dir, 'db', 'vim1')
 
    # get gabor features
    #get_stim_features(db_dir, feat_dir, 'train')
    # get candidate models
    #get_candidate_model(feat_dir, 'val')
    #get_candidate_model_new(db_dir, 'train')

    #-- general config
    subj_id = 1
    roi = 'v1'
    # directory config
    subj_dir = os.path.join(res_dir, 'vim1_S%s'%(subj_id))
    prf_dir = os.path.join(subj_dir, 'regress_prf')

    #-- pRF model fitting
    # pRF model tunning
    #get_vxl_idx(prf_dir, db_dir, subj_id, roi)
    #ridge_fitting(feat_dir, prf_dir, db_dir, subj_id, roi)
    #prf_selection(feat_dir, prf_dir, db_dir, subj_id, roi)
    #ridge_regression(prf_dir, db_dir, subj_id, roi)
    ridge_regression_model_test(prf_dir, db_dir, subj_id, roi)
    # get null distribution of tunning performance
    #null_distribution_prf_tunning(feat_dir, prf_dir, db_dir, subj_id, roi)
    # calculate tunning contribution of each gabor sub-banks
    #gabor_contribution2prf(feat_dir, prf_dir, db_dir, subj_id, roi)
    # pRF reconstruction
    #prf_recon(prf_dir, db_dir, subj_id, roi)
    # filter reconstruction
    #filter_recon(prf_dir, db_dir, subj_id, roi)
    # validation stimuli reconstruction
    #stimuli_recon(prf_dir, db_dir, subj_id, roi)
    # retinotopic mapping
    #retinotopic_mapping(prf_dir, roi)

