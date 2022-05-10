import time

import pyximport
from pyximport import install ; install()

import numpy
pyximport.install(setup_args={"include_dirs":numpy.get_include()},
                  reload_support=True)


import utils

from sktime.transformations.panel.OM_shapelet_transform_1.cython_shapelet_transform import RandomShapeletTransform


from sktime.datasets import load_unit_test  # This can be any of the baked in datasets

time1 = time.time()
X_train, y_train = load_unit_test(split="train", return_X_y=True)
t = RandomShapeletTransform(
    n_shapelet_samples = 500,
    max_shapelets = 10,
    batch_size = 100,)

t.fit(X_train, y_train)
X_t = t.transform(X_train)

time2 = time.time()

print("Total elapsed time: ", time2-time1)