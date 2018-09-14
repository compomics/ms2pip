from sys import argv
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy


model = argv[-1]
argv.remove(model)

setup(name='ms2pipfeatures_pyx_{}'.format(model),
      ext_modules=[Extension('cython_modules.ms2pipfeatures_pyx_{}'.format(model),
                             sources=['cython_modules/ms2pipfeatures_pyx_{}.pyx'.format(model)],
                             extra_compile_args=['-fno-var-tracking-assignments',
                                                 '-fno-var-tracking',
                                                 '-O3',
                                                 '-Wno-unused-result',
                                                 '-Wno-cpp',
                                                 '-Wno-unused-function'])],
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext})
