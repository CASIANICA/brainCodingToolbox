# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import ConfigParser
import tables
import numpy as np

class Config:
    def __init__(self, path):
        """Config instance initialization."""
        self.path = path
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(self.path)

    def get(self, field, key):
        """Get config value."""
        try:
            result = self.cf.get(field, key)
        except:
            result = ''
        return result

    def set(self, field, key, value):
        """Set config value."""
        try:
            self.cf.set(field, key, value)
            cf.write(open(self.path, 'w'))
        except:
            return False
        return True

def open_mat(mat_file_name, env='master'):
    """Open mat data file and retuen a File object.
    Argument `env` can be set as nica or mac for development."""
    cf = Config('config')
    if env=='master':
        db_dir = cf.get('base', 'master_path')
    else:
        db_dir = cf.get('base', 'dev_path')
    mat_file = os.path.join(db_dir, mat_file_name)
    f = tables.openFile(mat_file)
    return f
    ## show all variables available
    #f.listNodes
    #data = f.getNode('/rt')[:]
    #roi = f.getNode('/roi/v1lh')[:].flatten()
    #v1lh_idx = np.nonzero(roi==1)[0]
    #v1lh_resp = data[v1lh_idx]

