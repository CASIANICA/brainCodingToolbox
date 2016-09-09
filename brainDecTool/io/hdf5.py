# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import tables

def open_hdf5(mat_file):
    """Open hdf5 data file and retuen a File object."""
    f = tables.openFile(mat_file)
    return f

