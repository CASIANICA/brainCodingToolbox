# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

fig = plt.figure()

display_axes = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
# This is a nasty hack - using the hidden field to multiply the values
# such that 1 become 2*pi, this field is supposed to take values 1 or -1 only!!
display_axes._direction = 2*np.pi 
norm = mpl.colors.Normalize(0.0, 2*np.pi)
# Plot the colorbar onto the polar axis
# note - use orientation horizontal so that the gradient goes around
# the wheel rather than centre out
quant_steps = 2056
cb = mpl.colorbar.ColorbarBase(display_axes,
                               cmap=cm.get_cmap('RdBu_r', quant_steps),
                               norm=norm,
                               orientation='horizontal')

# aesthetics - get rid of border and axis labels 
cb.outline.set_visible(False) 
display_axes.set_axis_off()
# Replace with plt.savefig if you want to save a file
#plt.show()
plt.savefig('colorwheel.png')

