from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(name='ms2pipfeatures_pyx_ETD',
      ext_modules=[Extension('ms2pipfeatures_pyx_ETD',
                             sources=['ms2pipfeatures_pyx_ETD.pyx'],
                             extra_compile_args=['-fno-var-tracking-assignments',
                                                 '-fno-var-tracking',
                                                 '-O3',
                                                 '-Wno-unused-result',
                                                 '-Wno-cpp',
                                                 '-Wno-unused-function'])],
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext})
