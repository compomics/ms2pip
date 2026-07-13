# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `Spectrum` model: deprecated `model_validator`/`classmethod` combination, missing serializers for `np.ndarray` fields (broke `model_dump_json()`), and equality comparison crashing on array fields

## [4.2.0-beta.1] - 2026-06-19

### Added

- All prediction, m/z calculation, and spectrum annotation now use ms2rescore-rs (Rust); replaces C/Cython backend
- `ms2_tolerance_mode` parameter (`"Da"` or `"ppm"`) on all correlation/annotation functions and CLI
- `correlate` now accepts preloaded spectra on PSMs (`MS2Spectrum` / `AnnotatedMS2Spectrum`)
- `read_psms` accepts `list[PSM]` in addition to `PSMList`, `str`, and `Path`
- Replaced remaining C-based model files with native XGBoost models (HCD2019, TMT, and HCDch2)
- XGBoost models are now pre-loaded across `predict_library` batches for improved efficiency
- `rt` and `im` optional dependency groups for DeepLC and IM2Deep
- Min/max length validators on `ProteomeSearchSpace`
- CUDA_VISIBLE_DEVICES workaround on XGBoost model load (dmlc/xgboost#11283)
- API docs for `ms2pip.result` and `ms2pip.spectrum`

### Changed

- Minimum Python version raised to 3.11
- Package is now a pure-Python wheel (no compiled extensions)
- Batch processing uses Rayon-parallelized Rust calls instead of Python multiprocessing
- DeepLC integration uses `deeplc.predict()` functional API (replaces class-based `DeepLC` wrapper)
- IM2Deep integration uses `im2deep.predict()` functional API (replaces legacy `im2deep.im2deep.predict_ccs`)
- `predict_library` now runs RT/IM predictions once on the full filtered PSMList before batching (closes #243)
- Invalid peptidoforms (unsupported amino acids, length outside 4–100, missing charge) are skipped with a summarized warning instead of raising
- Bumped ms2rescore-rs requirement to `>=0.5.0a3,<2`
- Bumped psm_utils requirement to `>=1.5`

### Removed

- C/Cython components: `_cython_modules/`, `_models_c/`, `setup.py`, `MANIFEST.in`
- All Python-based multiprocessing (`_Parallelized` class)
- `_utils/retention_time.py` and `_utils/ion_mobility.py` (inlined into `core.py`)
- iRT calibration peptides (DeepLC v4 handles calibration internally)
- **Minor breaking change:** `ms2pip.constants.MODELS` entries no longer contain `id`, `peaks_version`, or `features_version` keys (C/Cython routing metadata); replaced by `fragmentation`. Code accessing these keys directly will break.

### Fixed

- `annotate-spectra` CLI crash (`.with_suffix()` called with no argument)
- `write_correlations` now accepts `Path` objects
- `_peptidoform_spaces` empty check uses `is None` instead of truthiness

## [4.1.2] - 2026-02-10

### Fixed

- Prefer (faster) Genesis for model downloads, with a fallback to Zenodo
- CI: Update build runners for macOS (see actions/runner-images#13046)
- Fix logging for spectral library prediction (by @paretje in #257)

## [4.1.1] - 2026-01-07

### Fixed

- Fix support for SQLAlchemy v2, keeping backwards compatibility with v1.4 (#249, fixes #250)
- Fix typo of `max_length` in `search_space` documentation (#245 by @paretje)
- CI: Do not build dependencies like pyarrow from source, which can cause failed build workflows

## [4.1.0] - 2025-01-20

### Added

- Support for Thermo raw (requiring dotnet runtime) and gzipped spectrum files, through mobiusklein/mzdata (#226, by @paretje)
- New Python API usage mode `correlate-single` to correlate a single `ObservedSpectrum` object with predictions (#232)
- Support for Python 3.12 and 3.13, NumPy v2, and newer pandas and XGBoost versions (#228)
- CI: Add integration test for `predict-single` (#228)

### Changed

- Ion mobility is now returned instead of collisional cross section when IM2Deep is used through the `add_ion_mobility` option (#236, by @rodvrees)
- Model files are now downloaded from Zenodo instead of Genesis (#225, by @paretje, fixes #229)
- Docs: Updated README to include `correlate-single`; moved webserver API docs from old Wiki (#239)

### Removed

- Removed support for Python 3.8 (EOL) and for musllinux distributions (#228, #237)
- Removed old conversion scripts, mostly implemented in psm_utils (#239)
- Removed old train script, to be replaced with a page in docs (#239)

## [4.0.0] - 2024-07-17

Fully refactored and substantially more user-friendly version of MS²PIP:

- More modular Python API
- One consolidated command-line interface with subcommands
- Support for all file formats readable by psm_utils
- Support for MGF, mzML, and Bruker raw spectrum files
- Support for ProForma 2.0 peptide notation (no modification configuration required)
- Support for multiple peptides/PSMs per spectrum
- Detailed documentation on [ms2pip.readthedocs.io](https://ms2pip.readthedocs.io)

## [3.13.0] - 2024-03-05

### Added

- New timsTOF 2024 model (now an alias for `timsTOF`) by @ArthurDeclercq in #211

## [3.12.0] - 2024-02-01

### Added

- Add code and figures for 2023 NAR manuscript (@ArthurDeclercq in #196 and #197)
- Backport timsTOF model from 4.0 dev release to 3.12 release (by @RalfG in #206)

## [3.11.0] - 2023-02-27

### Added

- `fasta2speclib`: Improved workflow for generating spectral libraries from a FASTA file, with new configuration options (PR #193, fixes #188)
  - Support for C-terminal modifications
  - Differentiate between peptide and protein termini for variable modifications
  - Allow filtering of peptides based on precursor m/z
  - Allow semi-specific cleavage
  - Allow non-specific cleavage
  - Allow setting of a maximum of variable modifications per peptide
  - Add tests for modification assignment
- Add figures for 2023 manuscript (PR #194)

### Changed

- Change logging of model configuration to debug level (PR #193)

### Removed

- `fasta2speclib`: Removed support for Elude-based RT predictions, RT predictions file, PEPREC filter, and saving temporary PEPREC files (PR #193)

### Fixed

- Remove unsupported argument for `mzml.read` (PR #193)
- `spectrum_output`: Fix CSV output to always use `\n` line terminators (PR #193)
- `spectrum_output`: Use semicolon separator for Spectronaut CSV output (PR #193)
- DeepLC integration: Disable PyGAM for default calibration on iRT peptides (PR #193)

## [3.10.1] - 2023-02-15

### Added

- If precursor charge is not found in the MGF file, the charge from the PeptideRecord file is used instead (#189)
- Add tests for fasta2speclib modification generation (#190)

### Fixed

- Fixed issue in `fasta2speclib` where fixed modifications were added one residue to the left of the actual site; bug introduced in v3.10.0 (#190)

## [3.10.0] - 2023-02-01

### Added

- Support for mzML spectrum files, both for evaluating models and for extracting feature vectors
- New argument `spectrum_id_pattern`: regular expression applied to spectrum titles before matching to peptide file entries
- When using MS²PIP as a class instance, the `pred_and_emp` dataframe can be returned instead of written to file by setting `return_results=True`
- If requested, retention time prediction with DeepLC is now also enabled when a spectrum file is provided

### Changed

- Improved logging: use Rich library for logging with timestamps and message log levels
- MS²PIP now shows a progress bar instead of verbose output during prediction
- `fasta2speclib`: Improved variable modification assignment algorithm; combinatorial explosion is now reduced by setting a maximum number of modified residues per peptide
- Switch to Pyteomics MGF reader
- Avoid SciPy dependency
- More optimal use of NumPy in `calc_correlations`

### Removed

- Removed unsupported Tableau output format

### Fixed

- Vastly improved computational speed and reduced memory usage when using XGBoost model files with a spectrum file input
- `fasta2speclib`: Fixed issue where modified versions of peptides were duplicated
- `spectrum_output`: Various fixes in MSP spectral library file writing for DIA-NN compatibility (m/z error of 0.0 per peak, modifications sorted by position, use `RetentionTime` instead of `RTINSECONDS`)
- Fixed `spectrum_utils` modification off-by-one bug (fixes #170)
- Updated `python_requires` to minimum 3.7

## [3.9.0] - 2022-03-12

### Added

- New CID-TMT prediction model for TMT-labelled peptide spectra acquired on ion trap (trap-type CID), for use in MultiNotch MS3 workflows (PR #157)
- Support for Python 3.9 and 3.10; dropped support for end-of-life Python 3.6 (PR #156, fixes #126)
- Support for alternative cleavage rules (digestion enzymes) in `fasta2speclib` (PR #166, fixes #96)
- `model_dir` option to set a custom directory for model downloads (CLI and Python API) (PR #169, fixes #165)
- Add docstring to `MS2PIP` class and example to `README.md` (PR #167)

### Changed

- Replaced C model files with XGBoost counterparts (except for HCD2019 and TMT): faster compilation, smaller package (PR #157)
- Various `fasta2speclib` improvements: add DeepLC option to default config, suppress TensorFlow warnings, replace deprecated `pandas.DataFrame.append` with `concat` (PR #166)
- Relaxed click version requirements (PR #157)
- Removed XGBoost warnings from CLI output (PR #157)

### Fixed

- Fixed missing support for XGBoost models in single-prediction mode (PR #157, fixes #155)

## [3.8.0] - 2021-11-14

### Added

- New models for non-tryptic peptides and immunopeptides (PR #137); see https://doi.org/10.1101/2021.11.02.466886
- Windows support (PR #151)
- Direct support for `.xgboost` model files; no dump to C and compilation required (PR #137)

### Fixed

- In DLIB output, a value is now written to the `isDecoy` column (#140, PR #152)

## [3.7.1] - 2021-09-13

### Fixed

- Pin NumPy version used during build to fix compatibility with older NumPy versions (PR #148)

## [3.7.0] - 2021-09-09

### Added

- New command to predict and plot a single spectrum (PR #136)

### Changed

- `fasta2speclib`: Pass through options from config file to DeepLC (PR #135, fixes #138)
- `fasta2speclib`: Pass `num_cpu` to DeepLC from the config

### Fixed

- Parse modifications on residue L (#144, PR #145)

## [3.6.3] - 2021-01-25

### Added

- Python 3.9 support (PR #122)
- bioconda package and biocontainers Docker image
- macOS support (PR #95, PR #127); not yet for Python 3.9 (#126)

### Fixed

- MS²PIP now exits on incorrectly configured or unknown modifications, instead of only showing a warning (#100, PR #101)
- Parsing of C-terminal modifications from a txt config file was broken in v3.6.2; now fixed (PR #109)
- Example `fasta2speclib` configuration erroneously contained average mass shifts; updated to monoisotopic mass shifts (PR #121)
- MS²PIP now exits with status code 1 on critical error (#102, PR #123)
- Supported config file extensions are now described in help and error messages (#125, PR #129)

## [3.6.2] - 2020-05-08

### Fixed

- Fixes in logging formatting (#64, #65)
- Use float formatting in CSV output
- Retention time predictions can be added without writing output to file
- When MS²PIP runs in a daemon process, it will not attempt to use multiprocessing
- Various improvements in `match_spectra` functionality (e.g. SQLite backend, output handling)
- General cleanup of repository (e.g. unused models)

## [3.6.1] - 2020-04-01

### Added

- New option `save_peprec` in `fasta2speclib` to save PEPREC files (including DeepLC predictions, if present)

### Fixed

- Small fix in `fasta2speclib` parameter handling

## [3.6.0] - 2020-03-30

### Added

- DeepLC integration for accurate LC retention time prediction, including for modified peptides; enable with the `-r` flag or `"add_retention_time": true` in `fasta2speclib` config
- Support for TOML-based configuration files
- New Skyline `.blib` to PEPREC and MGF converter script in `conversion_tools`

## [3.5.1] - 2020-03-04

### Fixed

- Add header files to manifest (hotfix)

## [3.5.0] - 2020-03-04

### Added

- `ProteinId` column added to Spectronaut CSV output

### Changed

- Major code refactoring
- Improved logging
- Improved exception handling
- Faster compilation

## [3.4.2] - 2019-12-31

### Added

- MS²PIP is now installable from PyPI with `pip install ms2pip` (no compilation step)
- MS²PIP now uses all available CPUs by default
- Moved to semantic versioning

### Changed

- Speed improvements after the multiprocessing step, especially for large numbers of predictions

## [v20191029] - 2019-10-29

### Added

- MS²PIP is now locally installable with pip and conda
- New Spectronaut CSV and Bibliospec/Skyline output formats
- Output formats can now be specified in the config file (e.g. `out=csv,msp,spectronaut`)
- Slightly faster model compilation

## [v20190624] - 2019-06-24

### Added

- Add `RetentionTimeMins` to MSP output when an ELUDE model file is provided

## [v20190312] - 2019-03-12

### Added

- Models for charge 2 fragment ions (CID and HCD)

## [v20190130] - 2019-01-30

### Fixed

- Bugfixes
- Updated requirements.txt

## [v20190120] - 2019-01-20

First official GitHub release of MS2PIPc (third iteration of MS2PIP).
