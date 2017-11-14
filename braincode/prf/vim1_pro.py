# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables
import bob.ip.gabor
from joblib import Parallel, delayed

from braincode.util import configParser
from braincode.math import make_2d_gaussian, ridge
from braincode.math.norm import zscore
from braincode.prf import dataio


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
    feat_ptr = []
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
        maxi = np.argmax(mcorr[:, i])
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

if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    # database directory config
    db_dir = os.path.join(cf.get('database', 'path'), 'vim1')
    # directory config for analysis
    root_dir = cf.get('base', 'path')
    feat_dir = os.path.join(root_dir, 'sfeatures', 'vim1')
    res_dir = os.path.join(root_dir, 'subjects')
 
    # get gabor features
    #get_stim_features(db_dir, feat_dir, 'train')
    # get candidate models
    #get_candidate_model(feat_dir, 'val')

    #-- general config
    subj_id = 1
    roi = 'v1'
    # directory config
    subj_dir = os.path.join(res_dir, 'vim1_S%s'%(subj_id))
    prf_dir = os.path.join(subj_dir, 'prf')

    #-- pRF model fitting
    # pRF model tunning
    ridge_fitting(feat_dir, prf_dir, db_dir, subj_id, roi)
    #prf_selection(feat_dir, prf_dir, db_dir, subj_id, roi)

