Prediction models
=================

Pre-trained MS²PIP models
-------------------------

MS²PIP includes multiple specialized prediction models, fit for peptide spectra
with different properties. These properties include fragmentation method,
instrument, labeling techniques and modifications. As all of these properties
can influence fragmentation patterns, it is important to match the MS²PIP model
to the properties of your experimental dataset.

All models are downloaded automatically upon first use. Model files can also be downloaded manually
from `genesis.ugent.be/uvpublicdata/ms2pip <https://genesis.ugent.be/uvpublicdata/ms2pip/>`_.

MS2 acquisition information and peptide properties of the models' training datasets:

+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| Model        | Fragmentation method | MS2 mass analyzer                      | Peptide properties                                 |
+==============+======================+========================================+====================================================+
| HCD2019      | HCD                  | Orbitrap                               | Tryptic digest                                     |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| HCD2021      | HCD                  | Orbitrap                               | Tryptic / Chymotrypsin digest                      |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| CID          | CID                  | Linear ion trap                        | Tryptic digest                                     |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| iTRAQ        | HCD                  | Orbitrap                               | Tryptic digest, iTRAQ-labeled                      |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| iTRAQphospho | HCD                  | Orbitrap                               | Tryptic digest, iTRAQ-labeled, enriched for        |
|              |                      |                                        | phosphorylation                                    |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| TMT          | HCD                  | Orbitrap                               | Tryptic digest, TMT-labeled                        |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| TTOF5600     | CID                  | Quadrupole time-of-flight              | Tryptic digest                                     |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| HCDch2       | HCD                  | Orbitrap                               | Tryptic digest                                     |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| CIDch2       | CID                  | Linear ion trap                        | Tryptic digest                                     |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| Immuno-HCD   | HCD                  | Orbitrap                               | Immunopeptides                                     |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| CID-TMT      | CID                  | Linear ion trap                        | Tryptic digest, TMT-labeled                        |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| timsTOF2023  | CID                  | Ion mobility quadrupole time-of-flight | Tryptic-, elastase digest, immuno class 1          |
+--------------+----------------------+----------------------------------------+----------------------------------------------------+
| timsTOF2024  | CID                  | Ion mobility quadrupole time-of-flight | Tryptic-, elastase digest, immuno class 1 & class 2|
+--------------+----------------------+----------------------------------------+----------------------------------------------------+

Models, version numbers, and the train and test datasets used to create each model:

+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| Model         | Current version   | Train-test dataset (unique peptides)            | Evaluation dataset (unique peptides)              | Median Pearson correlation on evaluation|
|               |                   |                                                 |                                                   | dataset                                 |
+===============+===================+=================================================+===================================================+=========================================+
| HCD2019       | v20190107         | `MassIVE-KB`_ (1 623 712)                       | `PXD008034`_ (35 269)                             | 0.903786                                |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| CID           | v20190107         | `NIST CID Human`_ (340 356)                     | `NIST CID Yeast`_ (92 609)                        | 0.904947                                |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| iTRAQ         | v20190107         | `NIST iTRAQ`_ (704 041)                         | `PXD001189`_ (41 502)                             | 0.905870                                |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| iTRAQphospho  | v20190107         | `NIST iTRAQ phospho`_ (183 383)                 | `PXD001189`_ (9 088)                              | 0.843898                                |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| TMT           | v20190107         | `Peng Lab TMT Spectral Library`_ (1 185 547)    | `PXD009495`_ (36 137)                             | 0.950460                                |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| TTOF5600      | v20190107         | `PXD000954`_ (215 713)                          | `PXD001587`_ (15 111)                             | 0.746823                                |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| HCDch2        | v20190107         | `MassIVE-KB`_ (1 623 712)                       | `PXD008034`_ (35 269)                             | 0.903786 (+)                            |
|               |                   |                                                 |                                                   | 0.644162 (++)                           |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| CIDch2        | v20190107         | `NIST CID Human`_ (340 356)                     | `NIST CID Yeast`_ (92 609)                        | 0.904947 (+)                            |
|               |                   |                                                 |                                                   | 0.813342 (++)                           |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| HCD2021       | v20210416         | Combined dataset (520 579)                      | `PXD008034`_ (35 269)                             | 0.932361                                |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| Immuno-HCD    | v20210316         | Combined dataset (460 191)                      | `PXD005231 (HLA-I)`_ (46 753)                     | 0.963736                                |
|               |                   |                                                 | `PXD020011 (HLA-II)`_ (23 941)                    |                                         |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| CID-TMT       | v20220104         | `PXD041002`_ (72 138)                           | `PXD005890`_ (69 768)                             | 0.851085                                |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| timsTOF2023   | v20230912         | Combined dataset (234 973)                      | `PXD043026` `PXD046535` `PXD046543`               | 0.892540 (tryptic)                      |
|               |                   |                                                 |                                                   | 0.871258 (elastase)                     |
|               |                   |                                                 |                                                   | 0.899834 (class I)                      |
|               |                   |                                                 |                                                   | 0.635548 (class II)                     |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+
| timsTOF2024   | v20240105         | Combined dataset (480 024)                      | `PXD043026` `PXD046535` `PXD046543` `PXD038782`   | 0.883270 (tryptic)                      |
|               |                   |                                                 |                                                   | 0.814374 (elastase)                     |
|               |                   |                                                 |                                                   | 0.887192 (class I)                      |
|               |                   |                                                 |                                                   | 0.847951 (class II)                     |
+---------------+-------------------+-------------------------------------------------+---------------------------------------------------+-----------------------------------------+

Training new MS²PIP models
--------------------------

[todo]


Prediction features
-------------------

The table below lists and describes all features generated and used by MS²PIP. These are mostly
based on four amino acid properties (basicity, hydrophobicity, helicity and isoelectric point)
for the full precursor and for the N- and C-terminal ions.

+-----------------+----------------------------------------------------------------------+
| Feature         | Description                                                          |
+=================+======================================================================+
| ``p_length``    | Precursor length                                                     |
+-----------------+----------------------------------------------------------------------+
| ``p_charge``    | Precursor charge                                                     |
+-----------------+----------------------------------------------------------------------+
| ``p_charge1``   | Precursor charge is 1 (one-hot encoding)                             |
+-----------------+----------------------------------------------------------------------+
| ``p_charge2``   | Precursor charge is 2 (one-hot encoding)                             |
+-----------------+----------------------------------------------------------------------+
| ``p_charge3``   | Precursor charge is 3 (one-hot encoding)                             |
+-----------------+----------------------------------------------------------------------+
| ``p_charge4``   | Precursor charge is 4 (one-hot encoding)                             |
+-----------------+----------------------------------------------------------------------+
| ``p_charge5``   | Precursor charge is 5 (one-hot encoding)                             |
+-----------------+----------------------------------------------------------------------+
| ``p_basi_min``  | Minimum basicity of the precursor                                    |
+-----------------+----------------------------------------------------------------------+
| ``p_basi_q1``   | First quartile of basicity of the precursor                          |
+-----------------+----------------------------------------------------------------------+
| ``p_basi_q2``   | Second quartile of basicity of the precursor                         |
+-----------------+----------------------------------------------------------------------+
| ``p_basi_q3``   | Third quartile of basicity of the precursor                          |
+-----------------+----------------------------------------------------------------------+
| ``p_basi_max``  | Maximum basicity of the precursor                                    |
+-----------------+----------------------------------------------------------------------+
| ``p_heli_min``  | Minimum helicity of the precursor                                    |
+-----------------+----------------------------------------------------------------------+
| ``p_heli_q1``   | First quartile of helicity of the precursor                          |
+-----------------+----------------------------------------------------------------------+
| ``p_heli_q2``   | Second quartile of helicity of the precursor                         |
+-----------------+----------------------------------------------------------------------+
| ``p_heli_q3``   | Third quartile of helicity of the precursor                          |
+-----------------+----------------------------------------------------------------------+
| ``p_heli_max``  | Maximum helicity of the precursor                                    |
+-----------------+----------------------------------------------------------------------+
| ``p_hydro_min`` | Minimum hydrophobicity of the precursor                              |
+-----------------+----------------------------------------------------------------------+
| ``p_hydro_q1``  | First quartile of hydrophobicity of the precursor                    |
+-----------------+----------------------------------------------------------------------+
| ``p_hydro_q2``  | Second quartile of hydrophobicity of the precursor                   |
+-----------------+----------------------------------------------------------------------+
| ``p_hydro_q3``  | Third quartile of hydrophobicity of the precursor                    |
+-----------------+----------------------------------------------------------------------+
| ``p_hydro_max`` | Maximum hydrophobicity of the precursor                              |
+-----------------+----------------------------------------------------------------------+
| ``p_iso_min``   | Minimum isoelectric point of the precursor                           |
+-----------------+----------------------------------------------------------------------+
| ``p_iso_q1``    | First quartile of isoelectric point of the precursor                 |
+-----------------+----------------------------------------------------------------------+
| ``p_iso_q2``    | Second quartile of isoelectric point of the precursor                |
+-----------------+----------------------------------------------------------------------+
| ``p_iso_q3``    | Third quartile of isoelectric point of the precursor                 |
+-----------------+----------------------------------------------------------------------+
| ``p_iso_max``   | Maximum isoelectric point of the precursor                           |
+-----------------+----------------------------------------------------------------------+
| ``n_length``    | Length of the N-terminal ion                                         |
+-----------------+----------------------------------------------------------------------+
| ``c_length``    | Length of the C-terminal ion                                         |
+-----------------+----------------------------------------------------------------------+
| ``n_count_A``   | Count of amino acid 'A' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_A``   | Count of amino acid 'A' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_C``   | Count of amino acid 'C' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_C``   | Count of amino acid 'C' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_D``   | Count of amino acid 'D' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_D``   | Count of amino acid 'D' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_E``   | Count of amino acid 'E' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_E``   | Count of amino acid 'E' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_F``   | Count of amino acid 'F' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_F``   | Count of amino acid 'F' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_G``   | Count of amino acid 'G' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_G``   | Count of amino acid 'G' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_H``   | Count of amino acid 'H' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_H``   | Count of amino acid 'H' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_I``   | Count of amino acid 'I' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_I``   | Count of amino acid 'I' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_K``   | Count of amino acid 'K' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_K``   | Count of amino acid 'K' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_M``   | Count of amino acid 'M' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_M``   | Count of amino acid 'M' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_N``   | Count of amino acid 'N' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_N``   | Count of amino acid 'N' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_P``   | Count of amino acid 'P' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_P``   | Count of amino acid 'P' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_Q``   | Count of amino acid 'Q' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_Q``   | Count of amino acid 'Q' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_R``   | Count of amino acid 'R' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_R``   | Count of amino acid 'R' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_S``   | Count of amino acid 'S' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_S``   | Count of amino acid 'S' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_T``   | Count of amino acid 'T' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_T``   | Count of amino acid 'T' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_V``   | Count of amino acid 'V' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_V``   | Count of amino acid 'V' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_W``   | Count of amino acid 'W' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_W``   | Count of amino acid 'W' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``n_count_Y``   | Count of amino acid 'Y' in the N-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``c_count_Y``   | Count of amino acid 'Y' in the C-terminal ion                        |
+-----------------+----------------------------------------------------------------------+
| ``p0_basi``     | basicity of the first amino acid of the peptide                      |
+-----------------+----------------------------------------------------------------------+
| ``p-1_basi``    | basicity of the last amino acid of the peptide                       |
+-----------------+----------------------------------------------------------------------+
| ``pi-1_basi``   | basicity of the amino acid before the fragmentation site             |
+-----------------+----------------------------------------------------------------------+
| ``pi_basi``     | basicity of the amino acid at the fragmentation site                 |
+-----------------+----------------------------------------------------------------------+
| ``pi+1_basi``   | basicity of the 1st amino acid after the fragmentation site          |
+-----------------+----------------------------------------------------------------------+
| ``pi+2_basi``   | basicity of the 2nd amino acid after the fragmentation site          |
+-----------------+----------------------------------------------------------------------+
| ``n_basi_sum``  | Sum of basicity of the N-terminal ion                                |
+-----------------+----------------------------------------------------------------------+
| ``n_basi_min``  | Minimum basicity of the N-terminal ion                               |
+-----------------+----------------------------------------------------------------------+
| ``n_basi_q1``   | First quartile of basicity of the N-terminal ion                     |
+-----------------+----------------------------------------------------------------------+
| ``n_basi_q2``   | Second quartile of basicity of the N-terminal ion                    |
+-----------------+----------------------------------------------------------------------+
| ``n_basi_q3``   | Third quartile of basicity of the N-terminal ion                     |
+-----------------+----------------------------------------------------------------------+
| ``n_basi_max``  | Maximum basicity of the N-terminal ion                               |
+-----------------+----------------------------------------------------------------------+
| ``c_basi_sum``  | Sum of basicity of the C-terminal ion                                |
+-----------------+----------------------------------------------------------------------+
| ``c_basi_min``  | Minimum basicity of the C-terminal ion                               |
+-----------------+----------------------------------------------------------------------+
| ``c_basi_q1``   | First quartile of basicity of the C-terminal ion                     |
+-----------------+----------------------------------------------------------------------+
| ``c_basi_q2``   | Second quartile of basicity of the C-terminal ion                    |
+-----------------+----------------------------------------------------------------------+
| ``c_basi_q3``   | Third quartile of basicity of the C-terminal ion                     |
+-----------------+----------------------------------------------------------------------+
| ``c_basi_max``  | Maximum basicity of the C-terminal ion                               |
+-----------------+----------------------------------------------------------------------+
| ``p0_heli``     | Helicity of the first amino acid of the peptide                      |
+-----------------+----------------------------------------------------------------------+
| ``p-1_heli``    | Helicity of the last amino acid of the peptide                       |
+-----------------+----------------------------------------------------------------------+
| ``pi-1_heli``   | Helicity of the amino acid before the fragmentation site             |
+-----------------+----------------------------------------------------------------------+
| ``pi_heli``     | Helicity of the amino acid at the fragmentation site                 |
+-----------------+----------------------------------------------------------------------+
| ``pi+1_heli``   | Helicity of the 1st amino acid after the fragmentation site          |
+-----------------+----------------------------------------------------------------------+
| ``pi+2_heli``   | Helicity of the 2nd amino acid after the fragmentation site          |
+-----------------+----------------------------------------------------------------------+
| ``n_heli_sum``  | Sum of helicity of the N-terminal ion                                |
+-----------------+----------------------------------------------------------------------+
| ``n_heli_min``  | Minimum helicity of the N-terminal ion                               |
+-----------------+----------------------------------------------------------------------+
| ``n_heli_q1``   | First quartile of helicity of the N-terminal ion                     |
+-----------------+----------------------------------------------------------------------+
| ``n_heli_q2``   | Second quartile of helicity of the N-terminal ion                    |
+-----------------+----------------------------------------------------------------------+
| ``n_heli_q3``   | Third quartile of helicity of the N-terminal ion                     |
+-----------------+----------------------------------------------------------------------+
| ``n_heli_max``  | Maximum helicity of the N-terminal ion                               |
+-----------------+----------------------------------------------------------------------+
| ``c_heli_sum``  | Sum of helicity of the C-terminal ion                                |
+-----------------+----------------------------------------------------------------------+
| ``c_heli_min``  | Minimum helicity of the C-terminal ion                               |
+-----------------+----------------------------------------------------------------------+
| ``c_heli_q1``   | First quartile of helicity of the C-terminal ion                     |
+-----------------+----------------------------------------------------------------------+
| ``c_heli_q2``   | Second quartile of helicity of the C-terminal ion                    |
+-----------------+----------------------------------------------------------------------+
| ``c_heli_q3``   | Third quartile of helicity of the C-terminal ion                     |
+-----------------+----------------------------------------------------------------------+
| ``c_heli_max``  | Maximum helicity of the C-terminal ion                               |
+-----------------+----------------------------------------------------------------------+
| ``p0_hydro``    | Hydrophobicity of the first amino acid of the peptide                |
+-----------------+----------------------------------------------------------------------+
| ``p-1_hydro``   | Hydrophobicity of the last amino acid of the peptide                 |
+-----------------+----------------------------------------------------------------------+
| ``pi-1_hydro``  | Hydrophobicity of the amino acid before the fragmentation site       |
+-----------------+----------------------------------------------------------------------+
| ``pi_hydro``    | Hydrophobicity of the amino acid at the fragmentation site           |
+-----------------+----------------------------------------------------------------------+
| ``pi+1_hydro``  | Hydrophobicity of the 1st amino acid after the fragmentation site    |
+-----------------+----------------------------------------------------------------------+
| ``pi+2_hydro``  | Hydrophobicity of the 2nd amino acid after the fragmentation site    |
+-----------------+----------------------------------------------------------------------+
| ``n_hydro_sum`` | Sum of hydrophobicity of the N-terminal ion                          |
+-----------------+----------------------------------------------------------------------+
| ``n_hydro_min`` | Minimum hydrophobicity of the N-terminal ion                         |
+-----------------+----------------------------------------------------------------------+
| ``n_hydro_q1``  | First quartile of hydrophobicity of the N-terminal ion               |
+-----------------+----------------------------------------------------------------------+
| ``n_hydro_q2``  | Second quartile of hydrophobicity of the N-terminal ion              |
+-----------------+----------------------------------------------------------------------+
| ``n_hydro_q3``  | Third quartile of hydrophobicity of the N-terminal ion               |
+-----------------+----------------------------------------------------------------------+
| ``n_hydro_max`` | Maximum hydrophobicity of the N-terminal ion                         |
+-----------------+----------------------------------------------------------------------+
| ``c_hydro_sum`` | Sum of hydrophobicity of the C-terminal ion                          |
+-----------------+----------------------------------------------------------------------+
| ``c_hydro_min`` | Minimum hydrophobicity of the C-terminal ion                         |
+-----------------+----------------------------------------------------------------------+
| ``c_hydro_q1``  | First quartile of hydrophobicity of the C-terminal ion               |
+-----------------+----------------------------------------------------------------------+
| ``c_hydro_q2``  | Second quartile of hydrophobicity of the C-terminal ion              |
+-----------------+----------------------------------------------------------------------+
| ``c_hydro_q3``  | Third quartile of hydrophobicity of the C-terminal ion               |
+-----------------+----------------------------------------------------------------------+
| ``c_hydro_max`` | Maximum hydrophobicity of the C-terminal ion                         |
+-----------------+----------------------------------------------------------------------+
| ``p0_iso``      | Isoelectric point of the first amino acid of the peptide             |
+-----------------+----------------------------------------------------------------------+
| ``p-1_iso``     | Isoelectric point of the last amino acid of the peptide              |
+-----------------+----------------------------------------------------------------------+
| ``pi-1_iso``    | Isoelectric point of the amino acid before the fragmentation site    |
+-----------------+----------------------------------------------------------------------+
| ``pi_iso``      | Isoelectric point of the amino acid at the fragmentation site        |
+-----------------+----------------------------------------------------------------------+
| ``pi+1_iso``    | Isoelectric point of the 1st amino acid after the fragmentation site |
+-----------------+----------------------------------------------------------------------+
| ``pi+2_iso``    | Isoelectric point of the 2nd amino acid after the fragmentation site |
+-----------------+----------------------------------------------------------------------+
| ``n_iso_sum``   | Sum of isoelectric points of the N-terminal ion                      |
+-----------------+----------------------------------------------------------------------+
| ``n_iso_min``   | Minimum isoelectric point of the N-terminal ion                      |
+-----------------+----------------------------------------------------------------------+
| ``n_iso_q1``    | First quartile of isoelectric points of the N-terminal ion           |
+-----------------+----------------------------------------------------------------------+
| ``n_iso_q2``    | Second quartile of isoelectric points of the N-terminal ion          |
+-----------------+----------------------------------------------------------------------+
| ``n_iso_q3``    | Third quartile of isoelectric points of the N-terminal ion           |
+-----------------+----------------------------------------------------------------------+
| ``n_iso_max``   | Maximum isoelectric point of the N-terminal ion                      |
+-----------------+----------------------------------------------------------------------+
| ``c_iso_sum``   | Sum of isoelectric points of the C-terminal ion                      |
+-----------------+----------------------------------------------------------------------+
| ``c_iso_min``   | Minimum isoelectric point of the C-terminal ion                      |
+-----------------+----------------------------------------------------------------------+
| ``c_iso_q1``    | First quartile of isoelectric points of the C-terminal ion           |
+-----------------+----------------------------------------------------------------------+
| ``c_iso_q2``    | Second quartile of isoelectric points of the C-terminal ion          |
+-----------------+----------------------------------------------------------------------+
| ``c_iso_q3``    | Third quartile of isoelectric points of the C-terminal ion           |
+-----------------+----------------------------------------------------------------------+
| ``c_iso_max``   | Maximum isoelectric point of the C-terminal ion                      |
+-----------------+----------------------------------------------------------------------+


.. _MassIVE-KB: https://doi.org/10.1016/j.cels.2018.08.004
.. _PXD008034: https://doi.org/10.1016/j.jprot.2017.12.006
.. _NIST CID Human: https://chemdata.nist.gov/
.. _NIST CID Yeast: https://chemdata.nist.gov/
.. _NIST iTRAQ: https://chemdata.nist.gov/
.. _PXD001189: https://doi.org/10.1182/blood-2016-05-714048
.. _NIST iTRAQ phospho: https://chemdata.nist.gov/
.. _PXD009495: https://doi.org/10.15252/msb.20188242
.. _Peng Lab TMT Spectral Library: https://doi.org/10.1021/acs.jproteome.8b00594
.. _PXD000954: https://doi.org/10.1038/sdata.2014.31
.. _PXD001587: https://doi.org/10.1038/nmeth.3255
.. _PXD005231 (HLA-I): https://doi.org/10.1101/098780
.. _PXD020011 (HLA-II): https://doi.org/10.3389/fimmu.2020.01981
.. _PXD041002: https://doi.org/10.1093/nar/gkad335
.. _PXD005890: https://doi.org/10.1021/acs.jproteome.7b00091
.. _Training new MS²PIP models: http://compomics.github.io/projects/ms2pip_c/wiki/Training-new-MS2PIP-models.html
