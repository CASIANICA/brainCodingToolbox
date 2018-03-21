# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import bob.ip.gabor
import skimage.measure
import tables

from braincode.util import configParser
from braincode.math import make_2d_gaussian


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

def get_gabor_kernels_small(feat_dir):
    """gabor bank generation.
    The smallest wavelength is 4 pixels (frequency 6.25 cyc/deg),
    and the largest is 96 pixels (frequency 0.26 cyc/deg).
    Each pixel indicates 0.04 degree of visual angle.
    """
    n_scale = 9
    gwt = bob.ip.gabor.Transform(number_of_scales=n_scale)
    gwt.generate_wavelets(500, 500)
    wavelets = {}
    for i in range(n_scale):
        s = int(2*4*np.sqrt(2)**i)
        smin = 250 - s/2
        wavelets['f%s_real'%(i+1)] = np.zeros((s, s, 8))
        wavelets['f%s_imag'%(i+1)] = np.zeros((s, s, 8))
        for j in range(8):
            w = bob.ip.gabor.Wavelet(resolution=(500, 500),
                            frequency=gwt.wavelet_frequencies[i*8+j])
            sw = bob.sp.ifft(w.wavelet.astype(np.complex128))
            preal = np.roll(np.roll(np.real(sw), 250, 0), 250, 1)
            pimag = np.roll(np.roll(np.imag(sw), 250, 0), 250, 1)
            wavelets['f%s_real'%(i+1)][...,j]=preal[smin:(smin+s),smin:(smin+s)]
            wavelets['f%s_imag'%(i+1)][...,j]=pimag[smin:(smin+s),smin:(smin+s)]
    outfile = os.path.join(feat_dir, 'gabor_kernels_small.npz')
    np.savez(outfile, **wavelets)

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

def get_vxl_coding_resp(feat_dir, prf_dir, rois):
    """Generate voxel-wise encoding model of specific roi."""
    if not isinstance(rois, list):
        rois = [rois]
    sel_vxl_idx = [6345,  6391,  6436,  4258,  4307,  6253]
    for roi in rois:
        roi_dir = os.path.join(prf_dir, roi)
        # load model parameters
        sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
        sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
        vxl_idx = np.load(os.path.join(roi_dir, 'vxl_idx.npy'))
        # load candidate model
        tmodels = np.load(os.path.join(feat_dir, 'train_candidate_model.npy'),
                          mmap_mode='r')
        # get voxel index in roi
        idxs = [i for i in range(vxl_idx.shape[0]) if vxl_idx[i] in sel_vxl_idx]
        for idx in idxs:
            print 'Voxel %s'%(vxl_idx[idx])
            model_idx = int(sel_models[idx])
            tx = np.array(tmodels[model_idx, ...]).astype(np.float64)
            m = np.mean(tx, axis=0, keepdims=True)
            s = np.std(tx, axis=0, keepdims=True)
            tx = (tx - m) / (s + 1e-5)
            wts = sel_paras[idx]
            pred = np.dot(tx, wts)
            print pred[:10]

def get_vxl_coding_wts(feat_dir, prf_dir, rois):
    """Generate voxel-wise encoding model of specific roi."""
    if not isinstance(rois, list):
        rois = [rois]
    # select voxels
    roi_vxl_idx = {}
    prf_center = np.zeros((500, 500))
    for roi in rois:
        roi_vxl_idx[roi] = []
        roi_dir = os.path.join(prf_dir, roi)
        # load model parameters 
        sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
        sel_model_corr = np.load(os.path.join(roi_dir,'reg_sel_model_corr.npy'))
        # threshold for voxel selection
        thr = 0.3
        rad = 200
        for i in range(sel_model_corr.shape[0]):
            if sel_model_corr[i] > thr:
                model_idx = int(sel_models[i])
                # get gaussian pooling field parameters
                si = model_idx / 2500
                sigma = [1] + [n*5 for n in range(1, 13)] + [70, 80, 90, 100]
                s = sigma[si]
                xi = (model_idx % 2500) / 50
                yi = (model_idx % 2500) % 50
                x0 = np.arange(5, 500, 10)[xi]
                y0 = np.arange(5, 500, 10)[yi]
                center_d = np.sqrt(np.square(x0-250)+np.square(y0-250))
                if (center_d<rad):
                    roi_vxl_idx[roi].append(i)
                    prf_center[y0, x0] = prf_center[y0, x0] + 1
    outdir = os.path.join(prf_dir,'tfrecon','_'.join(rois+[str(thr),str(rad)]))
    if not os.path.exists(outdir):
        os.makedirs(outdir, 0755)
    np.save(os.path.join(outdir, 'prf_center.npy'), prf_center)
    ## model test
    #for roi in rois:
    #    roi_vxl_idx[roi] = roi_vxl_idx[roi][:3]
    vxl_num = [len(roi_vxl_idx[roi]) for roi in roi_vxl_idx]
    print 'Selecte %s voxels'%(sum(vxl_num))

    # get voxel enoding wts
    masks = np.zeros((sum(vxl_num), 500, 500), dtype=np.float32)
    wts = np.zeros((sum(vxl_num), 72), dtype=np.float32)
    bias = np.zeros(sum(vxl_num))
    sel_vxl_idx = np.zeros(sum(vxl_num))
    # load model norm paras
    norm_paras = np.load(os.path.join(feat_dir, 'model_norm_paras.npz'))
    c = 0
    for roi in rois:
        # load model paras
        roi_dir = os.path.join(prf_dir, roi)
        sel_models = np.load(os.path.join(roi_dir, 'reg_sel_model.npy'))
        sel_paras = np.load(os.path.join(roi_dir, 'reg_sel_paras.npy'))
        vxl_idx = np.load(os.path.join(roi_dir, 'vxl_idx.npy'))
        for i in range(len(roi_vxl_idx[roi])):
            print 'ROI %s, Voxel %s'%(roi, roi_vxl_idx[roi][i])
            model_idx = int(sel_models[roi_vxl_idx[roi][i]])
            # get gaussian pooling field parameters
            si = model_idx / 2500
            xi = (model_idx % 2500) / 50
            yi = (model_idx % 2500) % 50
            x0 = np.arange(5, 500, 10)[xi]
            y0 = np.arange(5, 500, 10)[yi]
            sigma = [1] + [n*5 for n in range(1, 13)] + [70, 80, 90, 100]
            s = sigma[si]
            print 'center: %s, %s, sigma: %s'%(y0, x0, s)
            masks[c] = make_2d_gaussian(500, s, center=(x0, y0))
            vxl_wts = sel_paras[roi_vxl_idx[roi][i]]
            norm_mean = norm_paras['model_mean'][model_idx]
            norm_std = norm_paras['model_std'][model_idx]
            wts[c] = vxl_wts / norm_std 
            bias[c] = np.sum(vxl_wts * norm_mean / norm_std)
            sel_vxl_idx[c] = vxl_idx[roi_vxl_idx[roi][i]]
            c = c + 1
    outfile = os.path.join(outdir, 'vxl_coding_wts.npz')
    np.savez(outfile, vxl_idx=sel_vxl_idx, masks=masks, wts=wts, bias=bias)

def merge_stimuli(stim_dir):
    """Merge stimuli of training data from mat to numpy array."""
    stim = None
    for i in range(15):
        mat_file = os.path.join(stim_dir, 'Stimuli_Trn_FullRes_%02d.mat'%(i+1))
        mat = tables.open_file(mat_file)
        x = mat.get_node('/stimTrn')[:]
        mat.close()
        x = np.transpose(x, (1, 0, 2))
        if isinstance(stim, np.ndarray):
            stim = np.concatenate((stim, x), axis=2)
        else:
            stim = x
    np.save(os.path.join(stim_dir, 'train_stimuli.npy'), stim)

def merge_gabor_features(feat_dir):
    """Merge Gabor features of all images."""
    # get mean features from all images
    global_mean = np.zeros((1, 500, 500, 72))
    for i in range(14):
        tmp_file = os.path.join(feat_dir,
                        'Stimuli_Trn_FullRes_%02d_gabor_features.npy'%(i+1))
        tmp_data = np.load(tmp_file, mmap_mode='r')
        tmp_data = np.transpose(tmp_data, (0, 2, 3, 1))
        tmp_mean = np.mean(tmp_data, axis=0, keepdims=True)
        global_mean = global_mean + tmp_mean
    global_mean = global_mean / 14
    np.save(os.path.join(feat_dir, 'gabor_mean.npy'), global_mean)
    # merge gabor features
    outfile = os.path.join(feat_dir, 'train_stimuli_gabor_feat.memdat')
    fp = np.memmap(outfile, dtype='float32', mode='w+',
                   shape=(1750, 500, 500, 72))
    head_idx = 0
    for i in range(15):
        print 'Part %s'%(i+1)
        tmp_file = os.path.join(feat_dir,
                        'Stimuli_Trn_FullRes_%02d_gabor_features.npy'%(i+1))
        tmp_data = np.load(tmp_file)
        tmp_data = np.transpose(tmp_data, (0, 2, 3, 1))
        tmp_data = 10 * (tmp_data - global_mean)
        print tmp_data.max()
        print tmp_data.min()
        fp[head_idx:(head_idx+tmp_data.shape[0])] = tmp_data
        head_idx = head_idx + tmp_data.shape[0]


if __name__ == '__main__':
    """Main function."""
    # config parser
    cf = configParser.Config('config')
    # database directory config
    db_dir = os.path.join(cf.get('database', 'path'), 'vim1')
    stim_dir = os.path.join(db_dir, 'stimuli')
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
    get_gabor_kernels_small(feat_dir)
    #get_model_zparas(feat_dir)
    #get_vxl_coding_resp(feat_dir, prf_dir, ['v1', 'v2'])
    #get_vxl_coding_wts(feat_dir, prf_dir, ['v1', 'v2'])
    
    #merge_stimuli(stim_dir)
    #merge_gabor_features(feat_dir)

