# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import tables

from brainDecTool.io import hdf5 as bdio
from brainDecTool.util import configParser

# config parser
cf = configParser.Config('config')
data_dir = cf.get('base', 'path')

tf = tables.open_file(os.path.join(data_dir, 'VoxelResponses_subjecr1.mat'))
tf.listNodes
data = tf.get_node('/rt')[:]
roi = tf.get_node('/roi/v1lh')[:].flatten()
v1lh_idx = np.nonzero(roi==1)[0]
v1lh_resp = data[v1lh_idx]
tf.close()

