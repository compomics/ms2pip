# MS2PIPc extended instructions
## Prerequisites
Install the necessary Python3 modules. If Python3 is the only installed version of Python on your system, you can probably use `pip` instead of `pip3`
```
sudo pip3 install numpy pandas multiprocessing tables cython
```
To train models (instead of just running predictions with the pre-trained models), XGBoost is also required:
```
sudo pip3 install xgboost
```

## Install MS2PIPc
Download the repository from GitHub, go to the newly created project folder and compile the C-code:
```
git clone https://github.com/RalfG/ms2pip_c.git
cd ms2pip_c
sh compile.sh
```

## Configure MS2PIPc to your use case
A few parameters need to be set in the configuration file (by default `config.file`):
- `frag_method`: The peptide fragmentation method for which you want to predict spectra. This can be `HCD`, `CID` or `ETD`. The ETD models are still under heavy development and are not ready for general usage. Also, CID models are temporarily not included in the repository.
- `frag_error`: The MS/MS tolerance in Da. This defines the width of the region around theoretical fragment peak m/z's where MS2PIPc looks for emperical peaks. This value is only of importance if you provide MGF spectrum files.
- Post-translational modifications (PTMs): This part of the configuration file describes the PTMs that are present in the PEPREC file. Each line represents a certain PTM and is written as follows: `ptm=name,mass-shift,opt/fix,AA`. The name should be identical to the PTM name that is used in the PEPREC file and is case-sensitive. Next is the mass-shift of the PTM in Da. The next variable can be `opt` or `fix`, but is of no importance to MS2PIPc. Last in line is the one-letter code of the amino acid (AA) on which the PTM occurs. If a certain PTM occurs on different AAs, every AA should have it's own line in the configuration file and have a unique name (eg `PhosphoT`, `PhosphoS` and `PhosphoY`).
- N- and C-terminal modifications are described in the same way as normal PTMs, except for the fact that they do not have a specified AA. The syntax is as follows: `nterm=name,mass-shift,opt/fix` Again, the `opt`/`fix` parameter is not necessary for MS2PIPc.
- Lines can be commented out using a hash tag (`#`).

## Prepare your input data
MS2PIP always takes a PEPREC file as input. It is a space-separated file that lists all peptides. It has the following columns:
- `spec_id`: A unique ID for the peptide that, if an MGF file is given, matches with one fragmentation spectrum.
- `peptide`: Peptide sequence.
- `modifications`: PTMs for the given peptide. Every modification is listed as `name|location`, separated by a pipe (`|`) between the name, the location and other PTMs. The location is an integer counted starting at `1` for the first AA. `0` is reserved for N-terminal modifications. `Name` has to correspond to a PTM listed in the configuration file. Unmodified peptides are marked with a hyphen (`-`).
- `charge`: Precursor charge of the peptide.

Example of a PEPREC file:
```
spec_id modifications peptide charge
peptide1 - ACDE 2
peptide2 2|cam ACDEFGHI 3
peptide3 0|iTRAQ|3|Pyro_glu ACDEFGHIKMNPQ 2
```

In order to directly compare predictions to empirical spectra, or to create a feature-vector file to train new models, you will need to provide an MGF file. See the general `README.md` file for instructions on this.

## Run MS2PIPc
The basic MS2PIPc syntax is as follows:
```
python3 ms2pipC.py PEPREC
```

You can also add the following optional arguments:
```
  -h, --help      show help message and exit
  -c FILE         config file (by default config.file)
  -s FILE         .mgf MS2 spectrum file (optional)
  -w FILE         write feature vectors to FILE.{pkl,h5} (optional)
  -m INT          number of cpu's to use
```

If you want to use another configuration file, you need to give the file location with the `-c` argument. The command then becomes:
```
python3 ms2pipC.py -c CONFIG-FILE PEPREC
```

That's it! MS2PIPc should now be up and running. Instructions for more advanced use-cases (such as training your own models) can be found in the `README.md` file.
