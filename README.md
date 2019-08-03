# MS²PIP
[![GitHub release](https://img.shields.io/github/release-pre/compomics/ms2pip_c.svg)](https://github.com/compomics/ms2pip_c/releases)
[![Build Status](https://travis-ci.org/compomics/ms2pip_c.svg?branch=master)](https://travis-ci.org/compomics/ms2pip_c)
[![GitHub](https://img.shields.io/github/license/compomics/ms2pip_c.svg)](https://www.apache.org/licenses/LICENSE-2.0)

MS²PIP is a tool to predict MS² signal peak intensities from peptide sequences.
It employs the XGBoost machine learning algorithm and is written in Python.

You can install MS²PIP on your machine by following the [instructions below](https://github.com/compomics/ms2pip_c#installation) or the [extended install instructions](https://github.com/compomics/ms2pip_c/wiki/Extended_install_instructions).
For a more user friendly experience, we created a [web server](https://iomics.ugent.be/ms2pip)
. There, you can easily upload a list of peptide sequences, after which the
corresponding predicted MS² spectra can be downloaded in multiple file
formats. The web server can also be contacted through the
[RESTful API](https://iomics.ugent.be/ms2pip/api/).

If you use MS²PIP for your research, please cite the following articles:
- Gabriels, R., Martens, L., & Degroeve, S. (2019). Updated MS²PIP web server
delivers fast and accurate MS² peak intensity prediction for multiple
fragmentation methods, instruments and labeling techniques. Nucleic Acids
Research https://doi.org/10.1093/nar/gkz299
- Degroeve, S., Maddelein, D., & Martens, L. (2015). MS²PIP prediction server:
compute and visualize MS² peak intensity predictions for CID and HCD
fragmentation. Nucleic Acids Research, 43(W1), W326–W330.
https://doi.org/10.1093/nar/gkv542
- Degroeve, S., & Martens, L. (2013). MS²PIP: a tool for MS/MS peak intensity
prediction. Bioinformatics (Oxford, England), 29(24), 3199–203.
https://doi.org/10.1093/bioinformatics/btt544

Please also take note of and mention the MS²PIP-version and [model-version](#mspip-models) you used.

## Installation
Download the [latest release](https://github.com/compomics/ms2pip_c/releases/latest)
and unzip. MS2PIPc runs on Python 3.5 or greater. Build and install with Conda:
```
conda build . -c bioconda
conda install ms2pip --use-local
```
For development, use pip to install an editable version:
```
pip install --editable .
```

## Predicting MS2 peak intensities
MS2PIPc comes with pre-trained models for a variety of fragmentation methods and
modifications. These models can easily be applied by configuring MS2PIPc in the
[config.txt file](https://github.com/compomics/ms2pip_c#config-file) and
providing a list of peptides in the form of a [PEPREC file](https://github.com/compomics/ms2pip_c#peprec-file).

### MS2PIPc command line interface
```
usage: ms2pip [-h] [-c CONFIG_FILE] [-s MGF_FILE] [-w FEATURE_VECTOR_OUTPUT]
              [-t] [-m NUM_CPU]
              <PEPREC file>

positional arguments:
  <PEPREC file>             list of peptides

optional arguments:
  -h, --help                show this help message and exit
  -c CONFIG_FILE            config file (by default config.txt)
  -s MGF_FILE               .mgf MS2 spectrum file (optional)
  -w FEATURE_VECTOR_OUTPUT  write feature vectors to FILE.{pkl,h5} (optional)
  -t                        create Tableau Reader file
  -m NUM_CPU                number of cpu's to use
```

### Config file
Several MS2PIPc options need to be set in this config file.
- The models that should be used are set as `model=X` where X is one of the
currently supported MS²2PIP models (see [MS²PIP Models](#mspip-models)).
- The fragment ion error tolerance is set as `frag_error=X` where is X is the
tolerance in Da.
- PTMs (see further) are set as `ptm=X,Y,opt,Z` for each internal PTM where X is
a string that represents the PTM, Y is the difference in Da associated with the
PTM, opt is a required for compatibility with other CompOmics projects, and Z
is the amino acid IAA) that is modified by the PTM. For N- and C-terminal
modifications, Z should be `N-term` or `C-term`, respectively.


### Input files
#### PEPREC file
To apply the pre-trained models you need to pass *only* a `<PEPREC file>` to
MS2PIPc. This file contains the peptide sequences for which you want to predict
peak intensities. The file is space separated and contains at least the
following four columns:

- `spec_id`: unique id (string) for the peptide/spectrum. This must match the
TITLE field in the corresponding MGF file, if given.
- `modifications`: Amino acid modifications for the given peptide. Every
modification is listed as `location|name`, separated by a pipe (`|`) between the
location, the name, and other modifications. `location` is an integer counted
starting at `1` for the first AA. `0` is reserved for N-terminal modifications,
`-1` for C-terminal modifications. `name` has to correspond to a modification
listed in the [Config file](#config-file). Unmodified peptides are marked with
a hyphen (`-`).
- `peptide`: the unmodified amino acid sequence.
- `charge`: precursor charge state as an integer (without `+`).

Peptides must be strictly longer than 2 and shorter than 100 amino acids and
cannot contain the following amino acid one-letter codes: B, J, O, U, X or Z.
Peptides not fulfilling these requirements will be filtered out and will not be
reported in the output.

In the `conversion_tools` folder, we provide a host of Python scripts
to convert common search engine output files to a PEPREC file.


#### MGF file (optional)
Optionally, an MGF file with measured spectra can be passed to MS2PIPc. In this
case, MS2PIPc will calculate correlations between the measured and predicted
peak intensities. Make sure that the PEPREC `spec_id` matches the mgf `TITLE`
field. Spectra present in the MGF file, but missing in the PEPREC file (and
vice versa) will be skipped.

#### Example
Suppose the config file contains the following lines
```
ptm=Carbamidomethyl,57.02146,opt,C
ptm=Acetyl,42.010565,opt,N-term
ptm=Glyloss,-58.005479,opt,C-term
```
then the PEPREC file could look like this:
```
spec_id modifications peptide charge
peptide1 - ACDEK 2
peptide2 2|Carbamidomethyl ACDEFGR 3
peptide3 0|Acetyl|2|Carbamidomethyl ACDEFGHIK 2
```
In this example, `peptide3` is N-terminally acetylated and carries a
carbamidomethyl on its second amino acid.

The corresponding (optional) MGF file could contain the following spectrum:
```
BEGIN IONS
TITLE=peptide1
PEPMASS=283.11849750978325
CHARGE=2+
72.04434967 0.00419513
147.11276245 0.17418982
175.05354309 0.03652963
...
END IONS
```

### Output
The predictions are saved in a `.csv` file with the name
`<peptide_file>_predictions.csv`.
If you want the output to be in the form of an `.mgf` file, replace the
variable `mgf` in line 716 of `ms2pipC.py`.

### MS²PIP models
Currently the following models are supported in MS²PIP:
`HCD`, `CID`, `TTOF5600`, `TMT`, `iTRAQ`,
`iTRAQphospho`, `HCDch2` and `CIDch2`. The last two "ch2" models also include predictions for doubly charged fragment ions (b++ and y++), next to the predictions for singly charged b- and y-ions. 

If you use MS²PIP for your research, always mention the MS²PIP-version (see releases page) and model-version (see table below) you used.

#### Models, version numbers, and the train and test datasets used to create each model
Model | Current version | Train-test dataset (unique peptides) | Evaluation dataset (unique peptides) | Median Pearson correlation on evaluation dataset
-|-|-|-|-
HCD | v20190107 | [MassIVE-KB](https://doi.org/10.1016/j.cels.2018.08.004) (1 623 712) | [PXD008034](https://doi.org/10.1016/j.jprot.2017.12.006) (35 269) | 0.903786
CID | v20190107 | [NIST CID Human](https://chemdata.nist.gov/) (340 356) | [NIST CID Yeast](https://chemdata.nist.gov/) (92 609) | 0.904947
iTRAQ | v20190107 | [NIST iTRAQ](https://chemdata.nist.gov/) (704 041) | [PXD001189](https://doi.org/10.1182/blood-2016-05-714048) (41 502) | 0.905870
iTRAQphospho | v20190107 | [NIST iTRAQ phospho](https://chemdata.nist.gov/) (183 383) | [PXD001189](https://doi.org/10.1182/blood-2016-05-714048) (9 088) | 0.843898
TMT | v20190107 | [Peng Lab TMT Spectral Library](https://doi.org/10.1021/acs.jproteome.8b00594) (1 185 547) | [PXD009495](https://doi.org/10.15252/msb.20188242) (36 137) | 0.950460
TTOF5600 | v20190107 | [PXD000954](https://doi.org/10.1038/sdata.2014.31) (215 713) | [PXD001587](https://doi.org/10.1038/nmeth.3255) (15 111) | 0.746823
HCDch2 | v20190107 | [MassIVE-KB](https://doi.org/10.1016/j.cels.2018.08.004) (1 623 712) | [PXD008034](https://doi.org/10.1016/j.jprot.2017.12.006) (35 269) | 0.903786 (+) and 0.644162 (++)
CIDch2 | v20190107 | [NIST CID Human](https://chemdata.nist.gov/) (340 356) | [NIST CID Yeast](https://chemdata.nist.gov/) (92 609) | 0.904947 (+) and 0.813342 (++)

#### MS² acquisition information and peptide properties of the models' training datasets
For optimal results, your experimental data should match the properties of the MS²PIP model.

Model |	Fragmentation method	| MS² mass analyzer	| Peptide properties
-|-|-|-
HCD	| HCD	| Orbitrap |	Tryptic digest
CID |	CID	| Linear ion trap	| Tryptic digest
iTRAQ |	HCD	| Orbitrap |	Tryptic digest, iTRAQ-labeled
iTRAQphospho |	HCD |	Orbitrap |	Tryptic digest, iTRAQ-labeled, enriched for phosphorylation
TMT	| HCD	| Orbitrap	| Tryptic digest, TMT-labeled
TTOF5600 |	CID	| Quadrupole Time-of-Flight	| Tryptic digest
HCDch2	| HCD	| Orbitrap |	Tryptic digest
CIDch2 |	CID	| Linear ion trap	| Tryptic digest

To train custom MS2PIPc models, please refer to [Training new MS2PIP models](https://github.com/compomics/ms2pip_c/wiki/Training_new_MS2PIP_models) on our Wiki pages.
