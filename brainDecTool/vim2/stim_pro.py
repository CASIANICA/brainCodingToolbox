# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np

from brainDecTool.util import configParser
from brainDecTool.timeseries import hrf
import util as vutil

# config parser
cf = configParser.Config('config')
data_dir = cf.get('base', 'path')
stim_dir = os.path.join(data_dir, 'stimulus_val')

# load stimulus time courses
data = np.load(os.path.join(stim_dir, 'feat1_sti_val.npy'), mmap_mode='r')
# read time course from first node
# data.shape = (8100, 290400) for validation dataset.
tc = data[:, 0]
# convolve with HRF
# TODO: Caution of TIME RESOLUTION
hrf_signal = biGammaHRFSignal(20, 0.04);
con_tc = np.convolve(tc, hrf_signal)

