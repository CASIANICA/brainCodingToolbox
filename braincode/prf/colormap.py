# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import matplotlib as mpl
from matplotlib.colors import colorConverter


def get_overlay_colormap():
    color1 = colorConverter.to_rgba('red')
    color2 = colorConverter.to_rgba('yellow')
    maskcmp = mpl.colors.LinearSegmentedColormap.from_list('maskcolor',
                                                [color1, color2], 256)
    maskcmp._init()
    maskcmp._lut[:15, -1] = 0
    maskcmp._lut[15:, -1] = 0.45
    return maskcmp

def get_mask_colormap():
    color1 = colorConverter.to_rgba('yellow')
    color2 = colorConverter.to_rgba('red')
    maskcmp = mpl.colors.LinearSegmentedColormap.from_list('maskcolor',
                                                [color1, color2], 256)
    maskcmp._init()
    maskcmp._lut[:15, -1] = 0
    maskcmp._lut[15:, -1] = 0.65
    return maskcmp

