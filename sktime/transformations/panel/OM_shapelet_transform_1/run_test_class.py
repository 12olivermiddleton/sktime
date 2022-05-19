import time

import pyximport
from pyximport import install ; install()

import numpy
pyximport.install(setup_args={"include_dirs":numpy.get_include()},
                  reload_support=True)


# import utils

from sktime.transformations.panel.OM_shapelet_transform_1.cython_shapelet_transform import RandomShapeletTransform

from sktime.datasets import load_UCR_UEA_dataset  # This can be any of the baked in datasets
from sktime.datasets import tsc_dataset_names  # This can be any of the baked in datasets

datasets = tsc_dataset_names.univariate
working_datasets = []

for set in datasets:

    try:
        X_train, y_train = load_UCR_UEA_dataset(name=set, return_X_y=True)

    except:
        working_datasets.append(set)

print(working_datasets)




