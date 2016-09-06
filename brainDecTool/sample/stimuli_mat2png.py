# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
from scipy.misc import imsave

from brainDecTool.io import hdf5 as bdio
from brainDecTool.util import configParser

# config parser
cf = configParser.Config('config')
data_dir = cf.get('base', 'path')

tf = bdio.open_hdf5(os.path.join(data_dir, 'Stimuli.mat'))
#tf.listNodes
stimages = tf.get_node('/st')[:]
svimages = tf.get_node('/sv')[:]
tf.close()

# save training stimuli as png file
for i in range(stimages.shape[0]):
    x = stimages[i, :]
    new_x = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.uint8)
    new_x[..., 0] = x[0, ...]
    new_x[..., 1] = x[1, ...]
    new_x[..., 2] = x[2, ...]
    new_x = np.transpose(new_x, (1, 0, 2))
    file_name = 'train_' + str(i+1) + '.png'
    imsave(file_name, new_x)

# save validation stimuli as png file
for i in range(svimages.shape[0]):
    x = svimages[i, :]
    new_x = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.uint8)
    new_x[..., 0] = x[0, ...]
    new_x[..., 1] = x[1, ...]
    new_x[..., 2] = x[2, ...]
    new_x = np.transpose(new_x, (1, 0, 2))
    file_name = 'val_' + str(i+1) + '.png'
    imsave(file_name, new_x)
