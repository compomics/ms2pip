.. image:: https://github.com/compomics/ms2pip_c/raw/releases/img/ms2pip_logo_1000px.png
   :width: 150px
   :height: 150px

|

.. image:: https://img.shields.io/github/v/release/compomics/ms2pip_c?include_prereleases&style=flat-square
   :target: https://github.com/compomics/ms2pip_c/releases/latest/
.. image:: https://img.shields.io/pypi/v/ms2pip?style=flat-square
   :target: https://pypi.org/project/ms2pip/
.. image:: https://img.shields.io/github/actions/workflow/status/compomics/ms2pip_c/test.yml?branch=releases&label=tests&style=flat-square
   :target: https://github.com/compomics/ms2pip_c/actions/workflows/test.yml
.. image:: https://img.shields.io/github/actions/workflow/status/compomics/ms2pip_c/build_and_publish.yml?style=flat-square
   :target: https://github.com/compomics/ms2pip_c/actions/workflows/build_and_publish.yml
.. image:: https://img.shields.io/github/issues/compomics/ms2pip_c?style=flat-square
   :target: https://github.com/compomics/ms2pip_c/issues/
.. image:: https://img.shields.io/github/last-commit/compomics/ms2pip_c?style=flat-square
   :target: https://github.com/compomics/ms2pip_c/commits/releases/
.. image:: https://img.shields.io/github/license/compomics/ms2pip_c?style=flat-square
   :target: https://www.apache.org/licenses/LICENSE-2.0
.. image:: https://img.shields.io/twitter/follow/compomics?style=social
   :target: https://twitter.com/compomics

---------------------------------------------------------------------------------------------------

MS²PIP: MS2 Peak Intensity Prediction - Fast and accurate peptide fragmentation
spectrum prediction for multiple fragmentation methods, instruments and labeling techniques.

---------------------------------------------------------------------------------------------------

About
-----

MS²PIP is a tool to predict MS2 peak intensities from peptide sequences. The result is a predicted
peptide fragmentation spectrum that accurately resembles its observed equivalent. These predictions
can be used to validate peptide identifications, generate proteome-wide spectral libraries, or to
select discriminative transitions for targeted proteomics. MS²PIP employs the
`XGBoost <https://xgboost.readthedocs.io/en/stable/>`_ machine learning algorithm and is written in
Python and C.

.. figure:: https://raw.githubusercontent.com/compomics/ms2pip/v4.0.0/img/mirror-DVAQIFNNILR-2.png

   Mirror plot of an observed (top) and MS²PIP-predicted (bottom) spectrum for the peptide
   ``DVAQIFNNILR/2``.

You can install MS²PIP on your machine by following the
`installation instructions <https://ms2pip.readthedocs.io/en/latest/installation/>`_. For a more
user-friendly experience, go to the `MS²PIP web server <https://iomics.ugent.be/ms2pip>`_. There,
you can easily upload a list of peptide sequences, after which the corresponding predicted MS2
spectra can be downloaded in multiple file formats. The web server can also be contacted through
the `RESTful API <https://iomics.ugent.be/ms2pip/api/>`_.

The MS³PIP Python application can perform the following tasks:

- ``predict-single``: Predict fragmentation spectrum for a single peptide and optionally visualize
  the spectrum.
- ``predict-batch``: Predict fragmentation spectra for a batch of peptides.
- ``predict-library``: Predict a spectral library from protein FASTA file.
- ``correlate``: Compare predicted and observed intensities and optionally compute correlations.
- ``correlate-single``: Compare predicted and observed intensities for a single peptide spectrum.
- ``get-training-data``: Extract feature vectors and target intensities from observed spectra for
  training.
- ``annotate-spectra``: Annotate peaks in observed spectra.

MS²PIP supports a wide range of PSM input formats and spectrum output formats, and includes
pre-trained models for multiple fragmentation methods, instruments and labeling techniques. See
`Usage <https://ms2pip.readthedocs.io/en/latest/usage>`_ for more information.

Related projects
----------------

- `MS²Rescore <https://github.com/compomics/ms2rescore/>`_: Use MS²PIP and other peptide prediction
  tools to boost peptide identification results.
- `DeepLC <https://github.com/compomics/deeplc/>`_: Retention time prediction for (modified)
  peptides using deep learning.
- `IM2Deep <https://github.com/compomics/im2deep>`_: Ion mobility prediction for (modified)
  peptides using deep learning.
- `psm_utils <https://github.com/compomics/psm_utils/>`_: Common utilities for parsing and handling
  peptide-spectrum matches and search engine results in Python

Citations
---------

If you use MS²PIP for your research, please cite the following publication:

- Declercq, A., Bouwmeester, R., Chiva, C., Sabidó, E., Hirschler, A., Carapito, C., Martens, L.,
  Degroeve, S., Gabriels, R. (2023). Updated MS²PIP web server supports cutting-edge proteomics
  applications. `Nucleic Acids Research` `doi:10.1093/nar/gkad335 <https://doi.org/10.1093/nar/gkad335>`_

Prior MS²PIP publications:

- Gabriels, R., Martens, L., & Degroeve, S. (2019). Updated MS²PIP web server
  delivers fast and accurate MS2 peak intensity prediction for multiple
  fragmentation methods, instruments and labeling techniques. `Nucleic Acids
  Research` `doi:10.1093/nar/gkz299 <https://doi.org/10.1093/nar/gkz299>`_
- Degroeve, S., Maddelein, D., & Martens, L. (2015). MS²PIP prediction server:
  compute and visualize MS2 peak intensity predictions for CID and HCD
  fragmentation. `_Nucleic Acids Research`, 43(W1), W326–W330.
  `doi:10.1093/nar/gkv542 <https://doi.org/10.1093/nar/gkv542>`_
- Degroeve, S., & Martens, L. (2013). MS²PIP: a tool for MS/MS peak intensity
  prediction. `Bioinformatics (Oxford, England)`, 29(24), 3199–203.
  `doi:10.1093/bioinformatics/btt544 <https://doi.org/10.1093/bioinformatics/btt544>`_

Please also take note of, and mention, the MS²PIP version you used.

Full documentation
------------------

The full documentation, including installation instructions, usage examples,
and the command-line and Python API reference, can be found at
`ms2pip.readthedocs.io <https://ms2pip.readthedocs.io>`_.

Contributing
------------

Bugs, questions or suggestions? Feel free to post an issue in the
`issue tracker <https://github.com/compomics/ms2pip/issues/>`_ or to make a pull
request. Any contribution, small or large, is welcome!
