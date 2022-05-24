from setuptools import setup, Extension
# Create extension called utils, from file location
module = Extension('utils', sources=['sktime/transformations/panel/OM_shapelet_transform_1/utils.pyx'])
# Set up the extension passing environment and directories
setup(
    name='sktime',
    version='1.0',
    author='Oliver',
    include_dirs=["C:/Users/omidd/anaconda3/envs/sktime/lib/site-packages/numpy/core/include/"],
    ext_modules=[module]
)
