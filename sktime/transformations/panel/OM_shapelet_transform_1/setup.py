from distutils.core import setup
from Cython.Build import cythonize

setup(name="cython_shapelet_transform", ext_modules=cythonize('cython_shapelet_transform.pyx'))
