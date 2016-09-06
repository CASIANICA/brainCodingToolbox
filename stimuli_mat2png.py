# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
from scipy.misc import imsave
from brainDecodingToolbox import datasugar as sugar

base_dir = r'/home/huanglijie/workingdir/brainDecoding'

tf = sugar.open_mat(os.path.join(base_dir, 'Stimuli.mat'), env='dev')
#tf.listNodes
stimages = tf.get_node('/st')[:]
svimages = tf.get_node('/sv')[:]
tf.close()

# save training stimuli as png file
for i in stimages.shape[0]:
    x = stimages[i, :]
    new_x = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.uint8)
    new_x[..., 0] = x[0, ...]
    new_x[..., 1] = x[1, ...]
    new_x[..., 2] = x[2, ...]
    new_x = np.transpose(new_x, (1, 0, 2))
    file_name = 'train_' + str(i+1) + '.png'
    imsave(file_name, new_x)

# save validation stimuli as png file
for i in svimages.shape[0]:
    x = svimages[i, :]
    new_x = np.zeros((x.shape[1], x.shape[2], 3), dtype=np.uint8)
    new_x[..., 0] = x[0, ...]
    new_x[..., 1] = x[1, ...]
    new_x[..., 2] = x[2, ...]
    new_x = np.transpose(new_x, (1, 0, 2))
    file_name = 'val_' + str(i+1) + '.png'
    imsave(file_name, new_x)
