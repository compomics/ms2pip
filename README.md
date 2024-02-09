[<img src="https://github.com/compomics/ms2pip_c/raw/releases/img/ms2pip_logo_1000px.png" width="150" height="150" />](https://iomics.ugent.be/ms2pip/)
<br/><br/>

[![GitHub release](https://img.shields.io/github/v/release/compomics/ms2pip_c?include_prereleases&style=flat-square)](https://github.com/compomics/ms2pip_c/releases/latest/)
[![PyPI](https://img.shields.io/pypi/v/ms2pip?style=flat-square)](https://pypi.org/project/ms2pip/)
[![Tests](https://img.shields.io/github/actions/workflow/status/compomics/ms2pip_c/test.yml?branch=releases&label=tests&style=flat-square)](https://github.com/compomics/ms2pip_c/actions/workflows/test.yml)
[![Build](https://img.shields.io/github/actions/workflow/status/compomics/ms2pip_c/build_and_publish.yml?style=flat-square)](https://github.com/compomics/ms2pip_c/actions/workflows/build_and_publish.yml)
[![Open issues](https://img.shields.io/github/issues/compomics/ms2pip_c?style=flat-square)](https://github.com/compomics/ms2pip_c/issues/)
[![Last commit](https://img.shields.io/github/last-commit/compomics/ms2pip_c?style=flat-square)](https://github.com/compomics/ms2pip_c/commits/releases/)
[![GitHub](https://img.shields.io/github/license/compomics/ms2pip_c?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0)
[![Twitter](https://img.shields.io/twitter/follow/compomics?style=social)](https://twitter.com/compomics)

MS²PIP: MS² Peak Intensity Prediction - Fast and accurate peptide fragmention
spectrum prediction for multiple fragmentation methods, instruments and labeling techniques.

---

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Specialized prediction models](#specialized-prediction-models)

---

## Introduction
MS²PIP is a tool to predict MS² peak intensities from peptide sequences. The result is
a predicted peptide fragmentation spectrum that accurately resembles its observed
equivalent. These predictions can be used to validate peptide identifications, generate
proteome-wide spectral libraries, or to select discriminative transitions for targeted
proteomics. MS²PIP employs the XGBoost machine learning algorithm and is written in
Python.

You can install MS²PIP on your machine by following the
[installation instructions](#installation) below. For a more user-friendly experience,
go to the [MS²PIP web server](https://iomics.ugent.be/ms2pip). There, you can easily
upload a list of peptide sequences, after which the corresponding predicted MS² spectra
can be downloaded in multiple file formats. The web server can also be contacted
through the [RESTful API](https://iomics.ugent.be/ms2pip/api/).

To generate a predicted spectral library starting from a FASTA file, we
developed a pipeline called fasta2speclib. Usage of this pipeline is described
on the
[fasta2speclib wiki page](http://compomics.github.io/projects/ms2pip_c/wiki/fasta2speclib).
Fasta2speclib was developed in collaboration with the ProGenTomics group for the
[MS²PIP for DIA](https://github.com/brvpuyve/MS2PIP-for-DIA) project.

To improve the sensitivity of your peptide identification pipeline with MS²PIP
predictions, check out [MS²Rescore](https://github.com/compomics/ms2rescore/).

If you use MS²PIP for your research, please cite the following publication:
- Gabriels, R., Martens, L., & Degroeve, S. (2019). Updated MS²PIP web server
delivers fast and accurate MS² peak intensity prediction for multiple
fragmentation methods, instruments and labeling techniques. *Nucleic Acids
Research* [doi:10.1093/nar/gkz299](https://doi.org/10.1093/nar/gkz299)

Prior MS²PIP publications:
- Degroeve, S., Maddelein, D., & Martens, L. (2015). MS²PIP prediction server:
compute and visualize MS² peak intensity predictions for CID and HCD
fragmentation. *Nucleic Acids Research*, 43(W1), W326–W330.
[doi:10.1093/nar/gkv542](https://doi.org/10.1093/nar/gkv542)
- Degroeve, S., & Martens, L. (2013). MS²PIP: a tool for MS/MS peak intensity
prediction. *Bioinformatics (Oxford, England)*, 29(24), 3199–203.
[doi:10.1093/bioinformatics/btt544](https://doi.org/10.1093/bioinformatics/btt544)

Please also take note of, and mention, the MS²PIP version you used.

---

## Installation

[![install pip](https://flat.badgen.net/badge/install%20with/pip/green)](https://pypi.org/project/ms2pip/)
[![install bioconda](https://flat.badgen.net/badge/install%20with/bioconda/green)](https://bioconda.github.io/recipes/ms2pip/README.html)
[![container](https://flat.badgen.net/badge/pull/biocontainer/blue?icon=docker)](https://quay.io/repository/biocontainers/ms2pip)

#### Pip package

With Python 3.6 or higher, run:
```
pip install ms2pip
```

Compiled wheels are available for Python 3.6, 3.7, and 3.8, on 64bit Linux,
Windows, and macOS. This should install MS²PIP in a few seconds. For other
platforms, MS²PIP can be built from source, although it can take a while
to compile the large prediction models.

We recommend using a [venv](https://docs.python.org/3/library/venv.html) or
[conda](https://docs.conda.io/en/latest/) virtual environment.

#### Conda package

Install with activated bioconda and conda-forge channels:
```
conda install -c defaults -c bioconda -c conda-forge ms2pip
```

Bioconda packages are only available for Linux and macOS.

#### Docker container
First check the latest version tag on [biocontainers/ms2pip/tags](https://quay.io/repository/biocontainers/ms2pip?tab=tags). Then pull and run the container with
```
docker container run -v <working-directory>:/data -w /data quay.io/biocontainers/ms2pip:<tag> ms2pip <ms2pip-arguments>
```
where `<working-directory>` is the absolute path to the directory with your MS²PIP input files, `<tag>` is the container version tag, and `<ms2pip-arguments>` are the ms2pip command line options (see [Command line interface](#command-line-interface)).

#### For development

Clone this repository and use pip to install an editable version:
```
pip install --editable .
```

---

## Usage

1. [Fast prediction of large amounts of peptide spectra](#fast-prediction-of-large-amounts-of-peptide-spectra)
    1. [Command line interface](#command-line-interface)
    2. [Python API](#python-api)
    3. [Input files](#input-files)
        1. [Config file](#config-file)
        2. [PEPREC file](#peprec-file)
        3. [Spectrum file (optional)](#spectrum-file-optional)
        4. [Examples](#examples)
    4. [Output](#output)
2. [Predict and plot a single peptide spectrum](#predict-and-plot-a-single-peptide-spectrum)


### Fast prediction of large amounts of peptide spectra

MS²PIP comes with [pre-trained models](#specialized-prediction-models) for a
variety of fragmentation methods and modifications. These models can easily be
applied by configuring MS²PIP in the [config file](#config-file) and providing a
list of peptides in the form of a [PEPREC file](#peprec-file). Optionally,
MS²PIP predictions can be compared to observed spectra in an
[MGF or mzmL file](#spectrum-file-optional).

#### Command line interface

To predict a large amount of peptide spectra, use `ms2pip`:
```
usage: ms2pip [-h] -c CONFIG_FILE [-s SPECTRUM_FILE] [-w FEATURE_VECTOR_OUTPUT]
       [-r] [-x] [-m] [-t] [-n NUM_CPU]
       [--sqldb-uri SQLDB_URI]
       <PEPREC file>

positional arguments:
  <PEPREC file>         list of peptides

optional arguments:
  -h, --help            show this help message and exit
  -c, --config-file     Configuration file: text-based (extensions `.txt`,
                        `.config`, or `.ms2pip`) or TOML (extension `.toml`).
  -s, --spectrum-file   MGF or mzML spectrum file (optional)
  -w, --vector-file     write feature vectors to FILE.{pkl,h5} (optional)
  -r, --retention-time  add retention time predictions (requires DeepLC python package)
  -x, --correlations    calculate correlations (if spectrum file is given)
  -m, --match-spectra   match peptides to spectra based on predicted spectra (if spectrum file is given)
  -n, --num-cpu         number of CPUs to use (default: all available)
  --sqldb-uri           use sql database of observed spectra instead of spectrum files
  --model-dir           custom directory for downloaded XGBoost model files. By default, `~/.ms2pip` is used.
```

#### Python API

The `MS2PIP` class can be imported from `ms2pip.ms2pipC` and run as follows:
```python
>>> from ms2pip.ms2pipC import MS2PIP
>>> params = {
...     "ms2pip": {
...         "ptm": [
...             "Oxidation,15.994915,opt,M",
...             "Carbamidomethyl,57.021464,opt,C",
...             "Acetyl,42.010565,opt,N-term",
...         ],
...         "frag_method": "HCD",
...         "frag_error": 0.02,
...         "out": "csv",
...         "sptm": [], "gptm": [],
...     }
... }
>>> ms2pip = MS2PIP("test.peprec", params=params, return_results=True)
>>> predictions = ms2pip.run()
```

#### Input files
##### Config file
Several MS²PIP options need to be set in this config file.
- `model=X` where X is one of the currently supported MS²PIP models (see
[Specialized prediction models](#specialized-prediction-models)).
- `frag_error=X` where is X is the fragmentation spectrum mass tolerance in Da
(only relevant if a spectrum file is passed).
- `out=X` where X is a comma-separated list of a selection of the currently
supported output file formats: `csv`, `mgf`, `msp`, `spectronaut`, or
`bibliospec` (SSL/MS2, also for Skyline). For example: `out=csv,msp`.
- `ptm=X,Y,opt,Z` for every peptide modification where:
  - `X` is the PTM name and needs to match the names that are used in the
  [PEPREC file](#peprec-file)). If the `--retention_time` option is used, PTM names must
  match the PSI-MOD/Unimod names embedded in DeepLC (see
  [DeepLC documentation](https://github.com/compomics/DeepLC)).
  - `Y` is the mass shift in Da associated with the PTM.
  - `Z` is the one-letter code of the amino acid AA that is modified by the PTM.
For N- and C-terminal modifications, `Z` should be `N-term` or `C-term`,
respectively.

##### PEPREC file
To apply the pre-trained models you need to pass *only* a `<PEPREC file>` to
MS²PIP. This file contains the peptide sequences for which you want to predict
peak intensities. The file is space separated and contains at least the
following four columns:

- `spec_id`: unique id (string) for the peptide/spectrum. This must match the
`TITLE` field in the corresponding MGF file, or `nativeID` (MS:1000767) in the
corresponding mzML file, if given.
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

In the [conversion_tools](https://github.com/compomics/ms2pip_c/tree/releases/conversion_tools)
folder, we provide a host of Python scripts to convert common search engine
output files to a PEPREC file.

To start from a FASTA file, see [fasta2speclib](http://compomics.github.io/projects/ms2pip_c/wiki/fasta2speclib).


##### Spectrum file (optional)
Optionally, an MGF or mzML file with measured spectra can be passed to MS²PIP. In this
case, MS²PIP will calculate correlations between the measured and predicted
peak intensities. Make sure that the PEPREC `spec_id` matches the MGF `TITLE`
field or mzML `nativeID`. Spectra present in the spectrum file, but missing in the
PEPREC file (and vice versa) will be skipped.

##### Examples
Suppose the **config file** contains the following lines
```
model=HCD
frag_error=0.02
out=csv,mgf,msp
ptm=Carbamidomethyl,57.02146,opt,C
ptm=Acetyl,42.010565,opt,N-term
ptm=Glyloss,-58.005479,opt,C-term
```
then the **PEPREC file** could look like this:
```
spec_id modifications peptide charge
peptide1 - ACDEK 2
peptide2 2|Carbamidomethyl ACDEFGR 3
peptide3 0|Acetyl|2|Carbamidomethyl ACDEFGHIK 2
```
In this example, `peptide3` is N-terminally acetylated and carries a
carbamidomethyl on its second amino acid.

The corresponding (optional) **MGF file** can contain the following spectrum:
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

#### Output
The predictions are saved in the output file(s) specified in the
[config file](#config-file). Note that the normalization of intensities depends
on the output file format. In the CSV file output, intensities are
log2-transformed. To "unlog" the intensities, use the following formula:
`intensity = (2 ** log2_intensity) - 0.001`.


### Predict and plot a single peptide spectrum
With `ms2pip-single-prediction` a single peptide spectrum can be predicted with MS²PIP
and plotted with [spectrum_utils](https://spectrum-utils.readthedocs.io/). For instance,

```sh
ms2pip-single-prediction "PGAQANPYSR" "-" 3 --model TMT
```

results in:

![Predicted spectrum](img/PGAQANPYSR-3-TMT.png)

Run `ms2pip-single-prediction --help` for more details.

---

## Specialized prediction models
MS²PIP contains multiple specialized prediction models, fit for peptide spectra
with different properties. These properties include fragmentation method,
instrument, labeling techniques and modifications. As all of these properties
can influence fragmentation patterns, it is important to match the MS²PIP model
to the properties of your experimental dataset.

Currently the following models are supported in MS²PIP: `HCD`, `CID`, `iTRAQ`,
`iTRAQphospho`, `TMT`, `TTOF5600`, `HCDch2` and `CIDch2`. The last two "ch2"
models also include predictions for doubly charged fragment ions (b++ and y++),
next to the predictions for singly charged b- and y-ions.

### MS² acquisition information and peptide properties of the models' training datasets

| Model | Fragmentation method | MS² mass analyzer | Peptide properties |
| - | - | - | - |
| HCD2019 | HCD | Orbitrap | Tryptic digest |
| HCD2021 | HCD | Orbitrap | Tryptic/ Chymotrypsin digest |
| CID | CID | Linear ion trap | Tryptic digest |
| iTRAQ | HCD | Orbitrap | Tryptic digest, iTRAQ-labeled |
| iTRAQphospho | HCD | Orbitrap | Tryptic digest, iTRAQ-labeled, enriched for phosphorylation |
| TMT | HCD | Orbitrap | Tryptic digest, TMT-labeled |
| TTOF5600 | CID | Quadrupole Time-of-Flight | Tryptic digest |
| HCDch2 | HCD | Orbitrap | Tryptic digest |
| CIDch2 | CID | Linear ion trap | Tryptic digest |
| Immuno-HCD | HCD | Orbitrap | Immunopeptides |
| CID-TMT | CID | Linear ion trap | Tryptic digest, TMT-labeled |
| timsTOF2023 | CID | Ion mobility quadrupole time-of-flight | Tryptic-, elastase digest, immuno class 1 |
| timsTOF2024 | CID | Ion mobility quadrupole time-of-flight | Tryptic-, elastase digest, immuno class 1 & class 2 |

### Models, version numbers, and the train and test datasets used to create each model

| Model | Current version | Train-test dataset (unique peptides) | Evaluation dataset (unique peptides) | Median Pearson correlation on evaluation dataset |
| - | - | - | - | - |
| HCD2019 | v20190107 | [MassIVE-KB](https://doi.org/10.1016/j.cels.2018.08.004) (1 623 712) | [PXD008034](https://doi.org/10.1016/j.jprot.2017.12.006) (35 269) | 0.903786 |
| CID | v20190107 | [NIST CID Human](https://chemdata.nist.gov/) (340 356) | [NIST CID Yeast](https://chemdata.nist.gov/) (92 609) | 0.904947 |
| iTRAQ | v20190107 | [NIST iTRAQ](https://chemdata.nist.gov/) (704 041) | [PXD001189](https://doi.org/10.1182/blood-2016-05-714048) (41 502) | 0.905870 |
| iTRAQphospho | v20190107 | [NIST iTRAQ phospho](https://chemdata.nist.gov/) (183 383) | [PXD001189](https://doi.org/10.1182/blood-2016-05-714048) (9 088) | 0.843898 |
| TMT | v20190107 | [Peng Lab TMT Spectral Library](https://doi.org/10.1021/acs.jproteome.8b00594) (1 185 547) | [PXD009495](https://doi.org/10.15252/msb.20188242) (36 137) | 0.950460 |
| TTOF5600 | v20190107 | [PXD000954](https://doi.org/10.1038/sdata.2014.31) (215 713) | [PXD001587](https://doi.org/10.1038/nmeth.3255) (15 111) | 0.746823 |
| HCDch2 | v20190107 | [MassIVE-KB](https://doi.org/10.1016/j.cels.2018.08.004) (1 623 712) | [PXD008034](https://doi.org/10.1016/j.jprot.2017.12.006) (35 269) | 0.903786 (+) and 0.644162 (++) |
| CIDch2 | v20190107 | [NIST CID Human](https://chemdata.nist.gov/) (340 356) | [NIST CID Yeast](https://chemdata.nist.gov/) (92 609) | 0.904947 (+) and 0.813342 (++) |
| HCD2021 | v20210416 | [Combined dataset] (520 579) | [PXD008034](https://doi.org/10.1016/j.jprot.2017.12.006) (35 269)  | 0.932361
| Immuno-HCD | v20210316 | [Combined dataset] (460 191) | [PXD005231 (HLA-I)](https://doi.org/10.1101/098780) (46 753) <br>[PXD020011 (HLA-II)](https://doi.org/10.3389/fimmu.2020.01981 ) (23 941) | 0.963736<br>0.942383
| CID-TMT | v20220104 | [in-house dataset] (72 138) | [PXD005890](https://doi.org/10.1021/acs.jproteome.7b00091) (69 768) | 0.851085
| timsTOF2023 | v20230912 | [Combined dataset] (234 973) | PXD043026<br>PXD046535<br>PXD046543 (13 012) | 0.892540 (tryptic)<br>0.871258 (elastase)<br>0.899834 (class I)<br>0.635548 (class II)
| timsTOF2024 | v20240105 | [Combined dataset]  (480 024) | PXD043026<br>PXD046535<br>PXD046543<br>PXD038782 (25 265)  | 0.883270 (tryptic)<br>0.814374 (elastase)<br>0.887192 (class I)<br>0.847951 (class II)


To train custom MS²PIP models, please refer to [Training new MS²PIP models](http://compomics.github.io/projects/ms2pip_c/wiki/Training-new-MS2PIP-models.html) on our Wiki pages.
