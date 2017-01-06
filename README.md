#ms2pipXGB

###Install

Requirements:

- python numpy
- python pandas
- python multiprocessing
- python XGBoost (only required for training)
- Cython (http://cython.org/)

To compile the Cython `.pyx` files:

```
sh compile.sh
```

###Convert spectral library .msp to PEPREC format

The script

```
$ python convert_to_mgf.py <file>.msp
```

converts a spectral library in `.msp` format into a spectrum `.mgf` file and a peptide identification file in PEPREC format.
This format contains at least three columns: 

- `spec_id`: the id (TITLE) of the spectrum in the `.mgf` file
- `modifications`: a string indicating the locations (starting at 1) and the modification types
- `peptide`: the unmodified amino acid sequence

The files are written as `<file>.PEPREC` and `<file>.PEPREC.mgf`.

###Create feature vectors from PEPREC format

The script

```
usage: peprec2vec.py [-h] [-c INT] <.PEPREC file> <.PEPREC.mgf file>

MS2PIP on XGBoost

positional arguments:
  <.PEPREC file>        file containing peptide identifications
  <.PEPREC.mgf file>    file containing ms2 spectra

optional arguments:
  -h, --help            show this help message and exit
  -c INT, --num_cpu INT
                        number of cores
```

computes MS2PIP feature vectors from the PEPREC formatted files.
The following files are written:

- `vectors_b.pkl`: all feature vectors for the b-ions
- `vectors_y.pkl`: all feature vectors for the y-ions 
- `targets_b_1.pkl`: all targets (charge +1) values for the b-ions
- `targets_y_1.pkl`: all targets (charge +1) values for the y-ions
- `psmids.pkl`: groups the feature vectors by PSM

Currently (still under optimization) the following features are computed:

- `peplen`: number of amino acids in peptide
- `ionnumber`: number of amino acids in ion to predict
- `ionnumber_rel`: number of amino acids in ion to predict divided by `peplen`
- `pmz`: precursor mass
- `mean 'chem'`: mean of 'chem' in peptide
- `mz_ion`: mass of ion to predict
- `mz_ion_other`: mass of other ion
- `charge`: spectrum charge state
- `'chem'_'loc'`: `chem` value of amino acid at location 'loc' (nterm, cterm and relative to cleavage pos 'i')
- `mean_ion_'chem'`: mean of 'chem' in ion to predict
- `mean_ion_other_'chem'`: mean of 'chem' in other ion
- `min_ion_'chem'`: min value of 'chem' in ion to predict
- `min_ion_other_'chem'`: min value of 'chem' in other ion
- `max_ion_'chem'`: max value of 'chem' in ion to predict
- `max_ion_other_'chem'`: max value of 'chem' in other ion

The features `chem` are computed from tables with estimated chemical property values for basisity, hydrophobicity, helicity and pI.

###Optimize and Train XGBoost models

The script

```
usage: train_xgboost.py [-h] [-p INT]
                        <vectors.pkl> <targets.pkl> <psmids.pkl> <model type>

XGBoost training

positional arguments:
  <vectors.pkl>  feature vector file
  <targets.pkl>  target file
  <psmids.pkl>   PSM groups file
  <model type>   {b,y}

optional arguments:
  -h, --help     show this help message and exit
  -p INT         number of cpu's to use
```

reads the Pickle files written by `peprec2vec.py` and trains an XGBoost model. Hyperparameters should still be optimized.
You will need to digg into the script for model selection.

This script will also write the XGBoost models as `.c` files that can be compiled and linked through Cython. Please consult the script for details.
To compile the Cython links to the `.c` models just run the script `compile.sh` again.

###Run MS2PIP

The script

```
usage: ms2pipXGB.py [-h] [-c INT] <.PEPREC file> <.PEPREC.mgf file>

MS2PIP on XGBoost

positional arguments:
  <.PEPREC file>        file containing peptide identifications
  <.PEPREC.mgf file>    file containing ms2 spectra

optional arguments:
  -h, --help            show this help message and exit
  -c INT, --num_cpu INT
                        number of cores
```

will read the compiled `.c` models and predict the MS2 peak intensities for the `<.PEPREC.mgf>` file. These will be compared to the observed peak intensities computed from the 
`<.PEPREC>` file. The pearson correlation for the b and y-ion models are writted to the file `pearson.csv`.

