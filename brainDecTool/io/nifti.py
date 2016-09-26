# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import nibabel as nib

def save2nifti(data, file_name):
    """Save 3D/4D dataset as nifti file.
    Note that the header is derived from MNI standard template, thus the
    data should be in RAS space.
    FSL is required.
    """
    fsl_dir = os.getenv('FSLDIR')
    # for code testing
    if not fsl_dir:
        fsl_dir = r'/Users/sealhuang/repo/FreeROI/froi'
    template = os.path.join(fsl_dir, 'data', 'standard',
                            'MNI152_T1_2mm_brain.nii.gz')
    header = nib.load(template).header
    header['cal_max'] = data.max()
    header['cal_min'] = 0
    img = nib.Nifti1Image(data, None, header)
    nib.save(img, file_name)

