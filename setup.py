from subprocess import call
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy

call('rm -f cython_modules/ms2pip_pyx.c* cython_modules/ms2pip_pyx.so', shell=True)

setup(name='ms2pip_pyx',
      ext_modules=[Extension('cython_modules.ms2pip_pyx',
                             sources=['cython_modules/ms2pip_pyx.pyx'],
                             extra_compile_args=['-fno-var-tracking-assignments',
                                                 '-fno-var-tracking',
                                                 '-O3',
                                                 '-Wno-unused-result',
                                                 '-Wno-cpp',
                                                 '-Wno-unused-function'])],
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext})
