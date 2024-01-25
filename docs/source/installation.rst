Installation
============

Pip package
-----------
.. image:: https://flat.badgen.net/badge/install%20with/pip/green
   :target: https://pypi.org/project/ms2pip/

With Python 3.8 or higher, run:

.. code-block:: bash

   pip install ms2pip

Compiled wheels are available for various Python versions on 64bit Linux,
Windows, and macOS. This should install MS²PIP in a few seconds. For other
platforms, MS²PIP can be built from source, although it can take a while
to compile the large prediction models.

We recommend using a `venv <https://docs.python.org/3/library/venv.html>`__ or
`conda <https://docs.conda.io/en/latest/>`__ virtual environment.

Conda package
-------------
.. image:: https://flat.badgen.net/badge/install%20with/bioconda/green
   :target: https://bioconda.github.io/recipes/ms2pip/README.html

Install with activated bioconda and conda-forge channels:

.. code-block:: bash

   conda install -c defaults -c bioconda -c conda-forge ms2pip

Bioconda packages are only available for Linux and macOS.

Docker container
----------------
.. image:: https://flat.badgen.net/badge/pull/biocontainer/blue?icon=docker
   :target: https://quay.io/repository/biocontainers/ms2pip

First check the latest version tag on `biocontainers/ms2pip/tags <https://quay.io/repository/biocontainers/ms2pip?tab=tags>`__. Then pull and run the container with:

.. code-block:: bash

   docker container run -v <working-directory>:/data -w /data quay.io/biocontainers/ms2pip:<tag> ms2pip <ms2pip-arguments>

where `<working-directory>` is the absolute path to the directory with your MS²PIP input files, `<tag>` is the container version tag, and `<ms2pip-arguments>` are the ms2pip command line options (see :ref:`Command line interface`).

For development
---------------
Clone this repository and use pip to install an editable version:

.. code-block:: bash

   pip install --editable .


Optionally, add the ``[dev,docs]`` extras to install the development and
documentation dependencies:

.. code-block:: bash

   pip install --editable .[dev,docs]
