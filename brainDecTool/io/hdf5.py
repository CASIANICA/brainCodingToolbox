# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import tables

def open_hdf5(mat_file):
    """Open hdf5 data file and retuen a File object."""
    f = tables.openFile(mat_file)
    return f
    #data = f.getNode('/rt')[:]
    #roi = f.getNode('/roi/v1lh')[:].flatten()
    #v1lh_idx = np.nonzero(roi==1)[0]
    #v1lh_resp = data[v1lh_idx]

