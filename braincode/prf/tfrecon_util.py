# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import bob.ip.gabor
import skimage.measure

from braincode.util import configParser
from braincode.math import make_2d_gaussian
#from braincode.math import img_resize


def get_gabor_kernels(feat_dir):
    """gabor bank generation"""
    gwt = bob.ip.gabor.Transform(number_of_scales=9)
    gwt.generate_wavelets(500, 500)
    gabor_real = np.zeros((500, 500, 72))
    gabor_imag = np.zeros((500, 500, 72))
    for i in range(72):
        w = bob.ip.gabor.Wavelet(resolution=(500, 500),
                        frequency=gwt.wavelet_frequencies[i])
        sw = bob.sp.ifft(w.wavelet.astype(np.complex128)) 
        gabor_real[..., i] = np.roll(np.roll(np.real(sw), 250, 0), 250, 1)
        gabor_imag[..., i] = np.roll(np.roll(np.imag(sw), 250, 0), 250, 1)
    outfile = os.path.join(feat_dir, 'gabor_kernels.npz')
    np.savez(outfile, gabor_real=gabor_real, gabor_imag = gabor_imag)

def get_model_zparas(feat_dir):
    """Get mean and std of time courses for each model."""
    # load candidate models
    models = np.load(os.path.join(feat_dir, 'train_candidate_model.npy'),
                     mmap_mode='r')
    model_mean = np.zeros((42500, 72))
    model_std = np.zeros((42500, 72))
    for i in range(42500):
        print 'Model %s'%(i)
        x = np.array(models[i, ...]).astype(np.float64)
        model_mean[i] = np.mean(x, axis=0)
        model_std[i] = np.std(x, axis=0)
    outfile = os.path.join(feat_dir, 'model_norm_paras.npz')
    np.savez(outfile, model_mean=model_mean, model_std=model_std)

def get_vxl_coding_wts(feat_dir, prf_dir, roi):
    """Generate voxel-wise encoding model of specific roi."""
    roi_dir = os.path.join(prf_dir, roi)
    # load model parameters
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    # load model norm paras
    norm_paras = np.load(os.path.join(feat_dir, 'model_norm_paras.npz'))
    # select voxels
    thr = 0.3
    sel_vxl_idx = np.array([4, 5, 6])
    #sel_vxl_idx = np.nonzero(sel_model_corr>thr)[0]
    wt = np.zeros((250, 250, 72, sel_vxl_idx.shape[0]), dtype=np.float32)
    bias = np.zeros(sel_vxl_idx.shape[0])
    for i in range(sel_vxl_idx.shape[0]):
        print 'Voxel %s'%(sel_vxl_idx[i])
        model_idx = int(sel_models[sel_vxl_idx[i]])
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
        kernel = skimage.measure.block_reduce(kernel, (2, 2), np.mean)
        #kernel = np.expand_dims(kernel, 2)
        #kernel = img_resize(kernel, (250, 250))[..., 0]
        kernel = np.expand_dims(kernel, 0)
        kernel = np.repeat(kernel, 72, 0)
        coding_wts = sel_paras[sel_vxl_idx[i]]
        norm_mean = norm_paras['model_mean'][model_idx]
        norm_std = norm_paras['model_std'][model_idx]
        for c in range(72):
            kernel[c, ...] = kernel[c, ...] * coding_wts[c] / norm_std[c]
        kernel = np.swapaxes(kernel, 0, 1)
        kernel = np.swapaxes(kernel, 1, 2)
        wt[..., i] = kernel
        bias[i] = np.sum(coding_wts * norm_mean/2 / norm_std)
    outdir = os.path.join(roi_dir, 'tfrecon')
    if not os.path.exists(outdir):
        os.makedirs(outdir, 0755)
    outfile = os.path.join(outdir, 'vxl_coding_wts.npz')
    np.savez(outfile, wt=wt, bias=bias)

def get_vxl_coding_resp(feat_dir, prf_dir, roi):
    """Generate voxel-wise encoding model of specific roi."""
    roi_dir = os.path.join(prf_dir, roi)
    # load model parameters
    sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
    sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
    sel_model_corr = np.load(os.path.join(roi_dir, 'reg_sel_model_corr.npy'))
    # load candidate model
    tmodels = np.load(os.path.join(feat_dir, 'train_candidate_model.npy'),
                      mmap_mode='r')
    # select voxels
    thr = 0.3
    sel_vxl_idx = np.array([4, 5, 6])
    #sel_vxl_idx = np.nonzero(sel_model_corr>thr)[0]
    for i in range(sel_vxl_idx.shape[0]):
        print 'Voxel %s'%(sel_vxl_idx[i])
        model_idx = int(sel_models[sel_vxl_idx[i]])
        tx = np.array(tmodels[model_idx, ...]).astype(np.float64)
        m = np.mean(tx, axis=0, keepdims=True)
        s = np.std(tx, axis=0, keepdims=True)
        tx = (tx - m) / (s + 1e-5)
        wts = sel_paras[sel_vxl_idx[i]]
        pred = np.dot(tx, wts)
        print pred[:10]


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
    
    #-- general config
    subj_id = 1
    roi = 'v1'
    # directory config
    subj_dir = os.path.join(res_dir, 'vim1_S%s'%(subj_id))
    prf_dir = os.path.join(subj_dir, 'prf')
    
    #get_gabor_kernels(feat_dir)
    #get_model_zparas(feat_dir)
    get_vxl_coding_wts(feat_dir, prf_dir, roi)
    #get_vxl_coding_resp(feat_dir, prf_dir, roi)
