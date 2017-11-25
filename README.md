# MS2PIPc

### Install

Requirements:

- Python numpy
- Python pandas
- Python multiprocessing
- PyTables
- XGBoost (python API) (only required for training)
- Cython (http://cython.org/)

MS2PIPc requires the machine specific compilation of the C-code:

```
sh compile.sh
```


### MS2 peak intensity predictions

Pre-trained HCD models for the b- and y-ions can be found in
the `/models` folder. These C-coded decision tree models are compiled
by running the `compile.sh` script that writes the python module
`ms2pipfeatures_pyx.so` which is imported into the main python script
`ms2pipC.py`:  

```
usage: ms2pipC.py [-h] [-c FILE] [-s FILE] [-w FILE] [-m INT] <peptide file>

positional arguments:
  <peptide file>  list of peptides

optional arguments:
  -h, --help      show this help message and exit
  -c FILE         config file (by default config.file)
  -s FILE         .mgf MS2 spectrum file (optional)
  -w FILE         write feature vectors to FILE.{pkl,h5} (optional)
  -m INT          number of cpu's to use
```

The `-i` flag makes MS2PIPc use the NIST iTRAQ4 models (HCD only).

The `-i` flag in combination with the `-p` flag makes MS2PIPc use the NIST iTRAQ4 phospho models (HCD only).

### Config file (-c option)

Several MS2PIPc options need to be set in this config file.

The models that should be used are set as `frag_method=X` where X is either `CID`,  `HCD`, `ETD`, `HCDiTRAQ4` or `HCDiTRAQ4phospho`.
The fragment ion error tolerance is set as `frag_error=X` where is X is the tolerance in Da.

PTMs (see further) are set as `ptm=X,Y,o,Z` for each internal PTM where X is a string that represents
the PTM, Y is the difference in Da associated with the PTM, o is a field only used by Omega (can be any value) and Z is the amino
acid that is modified by the PTM. N-terminal modifications are specified as `nterm=X,Y,o`
where X is again a string that represents the PTM, o is a field only used by Omega (can be any value), and Y is again the difference in Da associated with the PTM.
Similarly, c-terminal modifications are specified as `cterm=X,Y,o`
where X is again a string that represents the PTM, o is a field only used by Omega (can be any value), and Y is again the difference in Da associated with the PTM.

### Getting predictions from peptide file

To apply the pre-trained models you need to pass *only*  a `<peptide file>`
to `ms2pipC.py`. This file contains the peptide sequences for which you
want to predict the b- and y-ion peak intensities. The file is space
separated and contains four columns with the following header names:

- `spec_id`: an id for the peptide/spectrum
- `modifications`: a string indicating the modified amino acids
- `peptide`: the unmodified amino acid sequence
- `charge`: charge state to predict

The predictions are saved in a `.csv` file with the name `<peptide_file>_predictions.csv`.
If you want the output to be in the form of an `.mgf` file, replace the variable
`mgf` in line 142 of `ms2pipC.py`.

The *spec_id* column is a unique identifier for each peptide that will
be used in the TITLE field of the predicted MS2 `.mgf` file. The
`modifications` column is a string that lists the PTMs in the peptide. Each PTM is written as
`A|B` where A is the location of the PTM in the peptide (the first amino acid has location 1,
location 0 is used for n-term
modifications, while -1 is used for c-term modifications) and B is a string that represent the PTM
as defined in the config file (`-c` command line argument).
Multiple PTMs in the `modifications` column are concatenated with '|'.
As an example, suppose the config file contains the line

```
ptm=Cam,57.02146,o,C
nterm=Ace,42.010565,o
cterm=Glyloss,-58.005479,o
```

then a modifications string could like `0|Ace|2|Cam|5|Cam|-1|Glyloss` which means that the second
and fifth amino acid is modified with `Cam`,  
that there is an N-terminal modification `Ace`,
and that there is a C-terminal modification `Glyloss`.

### Writing feature vectors for model training

To compile a feature vector dataset you need to supply the
MS2 .mgf file (option `-s`) and the name of the file to write the feature
vectors to (option `-w`) to `ms2pipC.py`.
The `spec_id` column in the `<peptide file>` should match the `TITLE` field
of the corresponding MS2 spectrum in the .mgf file and is used to find
the targets for the feature vectors.

#### Testing feature extraction
In the folder `tests`, run `pytest`. This will run the tests in
`test_features.py`, which verify if the feature and target extraction are
working properly. (The tests must be updated when we add or remove features!)
To do this the `pytest` package must be installed (`pip install pytest`)

### Convert spectral library .msp

The python script

```
$ python convert_to_mgf.py <file>.msp <title>
```

converts a spectral library in `.msp` format into a spectrum `.mgf` file,
 a `<peptide file>` and a `<meta>` file.


### Optimize and Train XGBoost models

The script

```
usage: train_xgboost_c.py [-h] [-c INT] [-t FILE] [-p] <_vectors.pkl> <type>

XGBoost training

positional arguments:
  <_vectors.pkl>  feature vector file
  <type>          model type: [B,Y,C,Z]

optional arguments:
  -h, --help      show this help message and exit
  -c INT          number of cpu's to use
  -t FILE         additional evaluation file
  -p              output plots
```

reads the pickled feature vector file `<vectors.pkl or .h5>` and trains an
XGBoost model. The `type` option should be `B` for b-ions, `Y` for
y-ions, `C` for c-ions and `Z` for z-ions.

Hyper parameters should still be optimized.
You will need to digg into the script for model selection.

This script will write the XGBoost models as `.c` files that can be compiled
and linked through Cython. Just put the models in the `/models` folder,
change the `#include` directives in `ms2pipfeatures_c.c`, and recompile
the `ms2pipfeatures_pyx.so` model by running the `compile.sh` script.
