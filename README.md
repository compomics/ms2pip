#MS2PIPC

###Install

Requirements:

- Python numpy
- Python pandas
- Python multiprocessing
- XGBoost (python API) (only required for training)
- Cython (http://cython.org/)

MS2PIPC requires the machine specific compilation of the C-code: 

```
sh compile.sh
```


###MS2 peak intensity predictions

Pre-trained HCD models for the b- and y-ions can be found in
the `/models` folder. These C-coded decision tree models are compiled 
by running the `compile.sh` script that writes the python module 
`ms2pipfeatures_pyx.so` which is imported into the main python script
`ms2pipC.py`:  

```
usage: ms2pipC.py [-h] [-s FILE] [-w FILE] [-c INT] <peptide file>

positional arguments:
  <peptide file>  list of peptides

optional arguments:
  -h, --help      show this help message and exit
  -s FILE         .mgf MS2 spectrum file (optional)
  -w FILE         write feature vectors to FILE_vectors.pkl (optional)
  -c INT          number of cpu's to use
```

To apply the pre-trained models you only need to pass a `<peptide file>`
to `ms2pipC.py`. This file contains the peptide sequences for which you
want to predict the b- and y-ion peak intensities. The file is space
separated and contains four columns with the following header names:

- `spec_id`: the id (TITLE) of the spectrum in the `.mgf` MS2 file
- `modifications`: a string indicating the modified amino acids
- `peptide`: the unmodified amino acid sequence
- `charge`: charge state to predict

The *spec_id* column is a unique identifier for each peptide that will
be used in the title field of the predicted MS2 .mgf file. The 
`modifications` column is a string that lists what amino acid positions
(starting with 1, position 0 is reserved for n-terminal modifications).
For instance the string "3|CAM|11|Oxidation" represents a Carbamidomethyl
modification of the 3th amino acid and a Oxidation modification of the
11th amino acid.

!! PREDICTION FROM PEPTIDE FILE ONLY IS NOT IMPLEMENTED YET !!
!! ONLY CAM and Oxidation are implemented (and CAM is considered fixed) !!

###Writing feature vectors for model training

To compile a (pickled) feature vector dataset you need to supply the 
MS2 .mgf file (option `-s`) and the name of the file to write the feature
vectors to (option `-w`) to `ms2pipC.py`.
The `spec_id` column in the `<peptide file>` should match the TITLE field
of the corresponding MS2 spectrum in the .mgf file and is used to find
the targets for the feature vectors.


###Convert spectral library .msp

The python script

```
$ python convert_to_mgf.py <file>.msp
```

converts a spectral library in `.msp` format into a spectrum `.mgf` file 
and a `<peptide file>`.

###Create feature vectors from PEPREC format

The script

```
usage: ms2pipC.py [-h] [-s FILE] [-w FILE] [-c INT] <peptide file>

positional arguments:
  <peptide file>  list of peptides

optional arguments:
  -h, --help      show this help message and exit
  -s FILE         .mgf MS2 spectrum file (optional)
  -w FILE         write feature vectors to FILE.pkl (optional)
  -c INT          number of cpu's to use
```

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

