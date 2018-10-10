# fasta2speclib configuration info

## fasta2speclib arguments
```
usage: fasta2speclib.py [-h] [-o OUTPUT_FILENAME] [-c CONFIG_FILENAME]
                        fasta_filename

Create an MS2PIP-predicted spectral library, starting from a fasta file.

positional arguments:
  fasta_filename      Path to the fasta file containing protein sequences

optional arguments:
  -h, --help          show this help message and exit
  -o OUTPUT_FILENAME  Name for output file(s) (if not given, derived from
                      input file)
  -c CONFIG_FILENAME  Name of configuration json file (default:
                      fasta2speclib_config.json)
```

## fasta2speclib_config.json
Name | Description | Possible values | Default value | Data type
--- | --- | --- | --- | ---
`output_filetype` | Output file formats for spectral library | `msp`, `mgf` and/or `hdf` | `["msp"]` | array of strings
`charges` | Precusor charges to include | positive integer | `[2, 3]` | array of numbers
`min_peplen` | Minimum length of peptides to include | positive integer | `8` | number
`max_pepmass` | Maximum peptide mass (in Dalton) to include | positive integer | `5000` | number
`missed_cleavages` | Number of missed cleavages to include | positive integer | `2` | number
`modifications` | Modifications to include. See below for more information | *see below* | modifications object | object
`ms2pip_model` | MS2PIP model to use for predictions | *see MS2PIPc documentation* | `"HCD"` | string
`decoy` | Also create decoy spectral library by reversing peptide sequences | `true`, `false` | `true` | boolean
`elude_model_file` | If not null, predict retention times with this ELUDE model* | `path/to/model.file` or `null` | `null` | string or null
`peprec_filter` | If not null, do not predict spectra for peptides present in this peprec | `path/to/peprec.file` or `null` | `null` | string or null
`batch_size` | To reduce memory consumption, the (still unmodified) peptides to predict are split-up into batches. A higher batch size is slightly faster, but requires more RAM. | positive integer | `5000` | number 
`num_cpu` | Number of processes for multithreading | positive integer | `24` | number

* For this functionality, ELUDE needs to be installed and callable with the command `elude`.

### Modifications
Modified versions of peptides can be included in the predicted spectral
library. For this, an array of modification objects needs to be entered into
the JSON config file. Every modification needs the following parameters:

Name | Description | Type
--- | --- | ---
`name` | Name of the modifications, as it will appear in the output files. Using the PSI-MS modification names is recommended. | string
`unimod_accession` | UniMod accession number (required for ELUDE RT predictions). | number
`mass_shift` | Mass shift the modification introduces. | number
`amino_acid` | Amino acid on which the modification occurs. If the modification does not occur on a specific amino acid (e.g. N-terminal acetylation), this can be set to `null`. | string or null
`n_term` | If the modification only occurs on the N-terminus, set to `true`. This can be combined with a specifically set amino_acid (e.g. Glu->pyro-Glu only occurs on N-terminal glutamic acid). | boolean
`fixed` | Set to `true` if only the modified version of the peptide should be present in the spectral library (e.g. for Carbamidomethyl). | boolean

Please take the following into account:
- As is the case in the MS2PIPc configuration, if a modification occurs on
multiple specific modifications (such as phosphorylation), a seperate entry is
required, each with a unique name (e.g. PhosphoS, PhosphoT and PhosphoY) for every
amino acid.
- If no modifications should be included in the spectral library, the
modifications object is an empty array (`[]`).
- C-terminal modifications are not yet supported!
- N-terminal modifications WITH specific first AA do not yet prevent other
modifications to be added on that first AA. This means that the function will,
for instance, combine Glu->pyro-Glu (combination of N-term and normal PTM) with
other PTMS for Glu on the first AA, while this is not possible in reality!