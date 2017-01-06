from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("ms2pipfeatures_pyx.pyx")
)
