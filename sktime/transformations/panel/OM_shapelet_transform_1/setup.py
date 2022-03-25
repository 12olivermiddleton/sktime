from setuptools import setup, Extension

module = Extension('utils', sources=['sktime/transformations/panel/OM_shapelet_transform_1/utils.pyx'])

setup(
    name='sktime',
    version='1.0',
    author='Oliver',
    ext_modules=[module]
)
