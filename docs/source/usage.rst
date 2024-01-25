#####
Usage
#####

Quickstart
==========

[todo]


Usage modes
===========

MS²PIP has various usage modes that each can be accessed through the command-line interface, or
through the Python API.

``predict-single``
------------------

In this mode, a single peptide spectrum is predicted with MS²PIP and optionally plotted with
`spectrum_utils <https://spectrum-utils.readthedocs.io/>`_. For instance,

.. code-block:: sh

    ms2pip predict-single "PGAQANPYSR/3" --model TMT

results in:

.. image:: ../../img/PGAQANPYSR-3-TMT.png
   :alt: Predicted spectrum


``predict-batch``
-----------------

[todo]

``predict-library``
-------------------

[todo]

This mode was first developed in collaboration with the ProGenTomics group for the
`MS²PIP for DIA <https://github.com/brvpuyve/MS2PIP-for-DIA>`_ project.

``correlate``
-------------

[todo]

``get-training-data``
---------------------

[todo]

To train a new prediction model, see :ref:`Training new MS²PIP models`.

``annotate-spectra``
---------------------

[todo]


Input
=====

Peptides / PSMs
---------------

PSM file types
~~~~~~~~~~~~~~

For peptide information input, MS²PIP accepts any file format that is supported by
:py:mod:`psm_utils`.See
`Supported file formats <https://psm-utils.readthedocs.io/en/stable/#supported-file-formats>`_ for
the full list. The simplest format is a tab-separated file with at least the columns
``peptidoform`` and ``spectrum_id`` present.

- ``peptidoform`` is the full
  `ProForma 2.0 notation <https://doi.org/10.1021/acs.jproteome.1c00771>`_ including amino acid
  modifications and precursor charge state.
- ``spectrum_id`` should match the ``TITLE`` or ``nativeID`` field of the related spectrum in the
  optional MGF or mzML file, if provided. Otherwise, any value is accepted.

For example:

.. code-block::

    peptidoform	spectrum_id
    RNVIM[Oxidation]DKVAK/2	1
    KHLEQHPK/2	2
    ...

See :py:mod:`psm_utils.io.tsv` for the full specification.

Peptide sequence properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Peptides must be strictly longer than 2 and shorter than 100 amino acids and
cannot contain the following amino acid one-letter codes: B, J, O, U, X or Z.
Peptides not fulfilling these requirements will be filtered out and will not be
reported in the output.

Amino acid modifications
~~~~~~~~~~~~~~~~~~~~~~~~

Amino acid modification labels must be resolvable to a known mass shift. This means that
accepted labels are:

- A name or accession from an controlled vocabulary, such as Unimod or PSI-MOD. (e.g.,
  ``Oxidation``, ``U:Oxidation``, ``U:35``, ``MOD:00046``...)
- An elemental formula (e.g, ``Formula:C12H20O2``)
- A mass shift in Da (e.g., ``+15.9949``)

Any unresolvable modification will result in an error. If needed, PSM files can be converted with
:py:mod:`psm_utils.io` and modifications can be renamed with the
:py:meth:`~psm_utils.psm_list.PSMList.rename_modifications()` method.

Spectrum file
-------------

In the :ref:`correlate` and :ref:`get-training-data` usage modes, an MGF or mzML file with observed
spectra must be provided to MS²PIP. Make sure that the PSM file ``spectrum_id`` matches the MGF
``TITLE`` field or mzML ``nativeID`` fields. Spectra present in the spectrum file, but missing in
the PSM file (and vice versa) will be skipped.


Output
======

The predictions are saved in the output file(s) specified command. Note that the normalization of
intensities depends on the output file format. In the CSV file output, intensities are
log2-transformed. To "unlog" the intensities, use the following formula:

.. code-block::

    intensity = (2 ** log2_intensity) - 0.001

