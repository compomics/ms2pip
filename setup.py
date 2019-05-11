from subprocess import call
from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy

call('rm -f ms2pip/cython_modules/ms2pip_pyx.c* ms2pip/cython_modules/ms2pip_pyx.so', shell=True)

extensions = [
    Extension(
        'ms2pip.cython_modules.ms2pip_pyx',
        sources=['ms2pip/cython_modules/ms2pip_pyx.pyx'],
        extra_compile_args=[
            '-fno-var-tracking-assignments',
            '-fno-var-tracking',
            '-O3',
            '-Wno-unused-result',
            '-Wno-cpp',
            '-Wno-unused-function'
        ]
    )
]

setup(
    name='ms2pip',
    version='20190510',
    description='MS²PIP: MS² Peak Intensity Prediction',
    author='Sven Degroeve, Ralf Gabriels, Ana Sílvia C. Silva',
    author_email='sven.degroeve@vib-ugent.be',
    url='https://www.github.com/compomics/ms2pip_c',
    packages=['ms2pip'],
    include_package_data=True,
    entry_points={
        'console_scripts': ['ms2pip=ms2pip.ms2pipC:main'],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Development Status :: 5 - Production/Stable"
    ],
    install_requires=[
        'numpy>=1.15.4',
        'Cython>=0.29.2',
        'pandas>=0.23.4',
        'tables>=3.4.4',
        'scipy>=1.1.0',
        'matplotlib>=3.0.2',
        'pyteomics>=4.1',
        'biopython>=1.72',
    ],
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    cmdclass={'build_ext': build_ext}
)
