from Cython.Build import build_ext
from numpy import get_include
from setuptools import setup
from setuptools.extension import Extension

extensions = [
    Extension(
        "ms2pip._cython_modules.ms2pip_pyx",
        sources=["ms2pip/_cython_modules/ms2pip_pyx.pyx"],
    )
]

setup(
    ext_modules=extensions,
    include_dirs=[get_include()],
    cmdclass={"build_ext": build_ext},
)
