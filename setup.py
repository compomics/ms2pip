import os
import platform
from glob import glob

import numpy
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension

# to_remove = [
#     "ms2pip/cython_modules/ms2pip_pyx.c*",
#     "ms2pip/cython_modules/ms2pip_pyx.so",
# ]
# _ = [[os.remove(f) for f in glob(pat)] for pat in to_remove]

# Large machine-written C model files require optimization to be disabled
compile_args = {
    "Linux": [
        "-O0",
        "-fno-var-tracking",
        "-Wno-unused-result",
        "-Wno-cpp",
        "-Wno-unused-function",
    ],
    "Darwin": [
        "-O0",
    ],
    "Windows": [
        "/Od",
        "/DEBUG",
        "/GL-",
        "/bigobj",
        "/wd4244",
    ],
}

extensions = [
    Extension(
        "ms2pip.cython_modules.ms2pip_pyx",
        sources=["ms2pip/cython_modules/ms2pip_pyx.pyx"] + glob("ms2pip/models_c/*/*.c"),
        extra_compile_args=compile_args[platform.system()],
    )
]

setup(
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    cmdclass={"build_ext": build_ext},
)
