# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def thist(time_series, bin=10):
    """Draw a histogram for input time series."""
    df = pd.DataFrame(time_series)
    df.hist(bins=bin)
    plt.show()

