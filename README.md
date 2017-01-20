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


###Optimize and Train XGBoost models

The script

```
usage: train_xgboost_c.py [-h] [-c INT] <vectors.pkl> <type>

XGBoost training

positional arguments:
  <_vectors.pkl>  feature vector file
  <type>         model type

optional arguments:
  -h, --help     show this help message and exit
  -c INT         number of cpu's to use
```

reads the pickled feature vector file `<_vectors.pkl>` and trains an 
XGBoost model. The `type` option should be "B" for b-ions and "Y" for 
y-ions.

Hyperparameters should still be optimized.
You will need to digg into the script for model selection.

This script will write the XGBoost models as `.c` files that can be compiled
and linked through Cython. Just put the models in the `/models` folder
, change the `#include` directives in `ms2pipfeatures_c.c`, and recompile 
the `ms2pipfeatures_pyx.so` model by running the `compile.sh` script. 
 
