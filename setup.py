import os
import shlex
from glob import glob

from setuptools import setup
from setuptools.command.build_clib import build_clib
from setuptools.command.develop import develop as st_develop
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import numpy

VERSION = "3.6.2"

NAME = "ms2pip"
LICENSE = "apache-2.0"
DESCRIPTION = "MS²PIP: MS² Peak Intensity Prediction"
AUTHOR = "Sven Degroeve, Ralf Gabriels, Kevin Velghe, Ana Sílvia C. Silva"
AUTHOR_EMAIL = "sven.degroeve@vib-ugent.be"
URL = "https://www.github.com/compomics/ms2pip_c"
PROJECT_URLS = {
    "Documentation": "http://compomics.github.io/projects/ms2pip_c",
    "Source": "https://github.com/compomics/ms2pip_c",
    "Tracker": "https://github.com/compomics/ms2pip_c/issues",
    "Webserver": "https://iomics.ugent.be/ms2pip/",
    "Publication": "https://doi.org/10.1093/nar/gkz299/",
}
KEYWORDS = [
    "MS2PIP",
    "Proteomics",
    "peptides",
    "peak intensity prediction",
    "spectrum",
    "machine learning",
    "spectral library",
    "fasta2speclib",
]
CLASSIFIERS = [
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: POSIX :: Linux",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Development Status :: 5 - Production/Stable",
]
INSTALL_REQUIRES = [
    "biopython>=1.74,<2",
    "numpy>=1.16,<2",
    "pandas>=0.24,<2",
    "pyteomics>=3.5,<5",
    "scipy>=1.2,<2",
    "tqdm>=4,<5",
    "tables>=3.4",
    "tomlkit>=0.5.11,<1"
]
PYTHON_REQUIRES = ">=3.6,<4"

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

extra_compile_args = [
        "-fno-var-tracking",
        '-Wno-unused-result',
        '-Wno-cpp',
        '-Wno-unused-function',
    ]

libms2pip = ('ms2pip', {'sources': ["ms2pip/cython_modules/init.c", "ms2pip/cython_modules/features.c", "ms2pip/cython_modules/peaks.c"] + glob("ms2pip/models/*/*.c"), 'cflags': extra_compile_args})

if 'CFLAGS' in os.environ:
    cflags = shlex.split(os.environ['CFLAGS'])
    if '-UNDEBUG' not in cflags:
        extra_compile_args.append('-DCYTHON_WITHOUT_ASSERTIONS')

extensions = [
    Extension(
        "ms2pip.cython_modules.ms2pip_pyx",
        sources=["ms2pip/cython_modules/ms2pip_pyx.pyx"],
        extra_compile_args=extra_compile_args,
    )
]


class develop(st_develop):
    __doc__ = st_develop.__doc__

    def install_for_development(self):
        self.run_command('build_clib')
        st_develop.install_for_development(self)


setup(
    name=NAME,
    version=VERSION,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    keywords=KEYWORDS,
    classifiers=CLASSIFIERS,
    packages=["ms2pip", "ms2pip.ms2pip_tools", "fasta2speclib"],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "ms2pip=ms2pip.__main__:main",
            "fasta2speclib=fasta2speclib.fasta2speclib:main",
        ],
    },
    install_requires=INSTALL_REQUIRES,
    python_requires=PYTHON_REQUIRES,
    libraries=[libms2pip],
    ext_modules=extensions,
    include_dirs=[numpy.get_include()],
    # library_dirs=[os.path.join(os.path.dirname(__file__), 'ms2pip', 'cython_modules')],
    cmdclass={"build_clib": build_clib, "build_ext": build_ext, 'develop': develop},
)
