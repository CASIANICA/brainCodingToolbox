import numpy as np
from matplotlib.mlab import PCA

data = np.array(np.random.randint(10, size=(10, 3)))
res = PCA(data)
