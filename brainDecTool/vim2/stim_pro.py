# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np

from brainDecTool.util import configParser
from brainDecTool.timeseries import hrf

# config parser
cf = configParser.Config('config')
data_dir = cf.get('base', 'path')
stim_dir = os.path.join(data_dir, 'cnn_rsp')

# scanning parameter
TR = 1
# movie fps
fps = 15
time_unit = 1.0 / fps

# HRF config
hrf_times = np.arange(0, 35, time_unit)
hrf_signal = hrf.biGammaHRF(hrf_times)

def get_val_tr():
    """Get TRs from validateion dataset."""
    # load stimulus time courses
    feat1_ts = np.load(os.path.join(stim_dir, 'feat1_sti_val.npy'),
                       mmap_mode='r')
    # data.shape = (8100, 290400) for validation dataset.
    ts_shape = feat1_ts.shape

    # data array for storing time series after convolution and down-sampling
    feat = np.zeros((ts_shape[1], ts_shape[0]/fps))
    # convolution and down-sampling
    for i in range(ts_shape[1]):
        ts = feat1_ts[:, i]
        # log-transform
        ts = np.log(ts+1)
        # convolved with HRF
        convolved = np.convolve(ts, hrf_signal)
        # remove time points after the end of the scanning run
        n_to_remove = len(hrf_times) - 1
        convolved = convolved[:-n_to_remove]
        # down-sampling
        vol_times = np.arange(0, ts_shape[0], fps)
        feat[i, :] = convolved[vol_times]

    # save data
    out_file = os.path.join(stim_dir, 'feat1_val_trs.npy')
    np.save(out_file, feat)

def get_train_tr():
    """Get TRs from training dataset."""
    # load stimulus time courses
    cnn_dir = r'/media/DATA78/stimulus_maxuelin/stimulus_train'
    feat1_ptr = []
    time_count = 0
    for i in range(12):
        tmp = np.load(os.path.join(cnn_dir, 'feat1_sti_train_'+str(i+1)+'.npy'),
                      mmap_mode='r')
        time_count += tmp.shape[0]
        feat1_ptr.append(tmp)

    # data array for storing time series after convolution and down-sampling
    feat = np.zeros((feat1_ptr[0].shape[1], time_count/fps))
    # convolution and down-sampling
    for i in range(feat1_ptr[0].shape[1]):
        for p in range(12):
            if not p:
                ts = feat1_ptr[p][:, i]
            else:
                ts = np.concatenate([ts, feat1_ptr[p][:, i]])
        print ts.shape
        # log-transform
        ts = np.log(ts+1)
        # convolved with HRF
        convolved = np.convolve(ts, hrf_signal)
        # remove time points after the end of the scanning run
        n_to_remove = len(hrf_times) - 1
        convolved = convolved[:-n_to_remove]
        # down-sampling
        vol_times = np.arange(0, time_count, fps)
        feat[i, :] = convolved[vol_times]

    # save data
    out_file = os.path.join(stim_dir, 'feat1_train_trs.npy')
    np.save(out_file, feat)
    

if __name__ == "__main__":
    get_train_tr()

