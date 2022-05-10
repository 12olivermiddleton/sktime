import os

import numpy
from setuptools import setup, Extension

module = Extension('utils', sources=['sktime/transformations/panel/OM_shapelet_transform_1/utils.pyx'])

# os.environ["C_INCLUDE_PATH"] = numpy.get_include()
# C:\Users\omidd\anaconda3\envs\sktime\lib\site-packages\numpy\core\include

setup(
    name='sktime',
    version='1.0',
    author='Oliver',
    include_dirs=["C:/Users/omidd/anaconda3/envs/sktime/lib/site-packages/numpy/core/include/"],
    ext_modules=[module]
)
