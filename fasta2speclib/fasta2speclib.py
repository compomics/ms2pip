"""
Create a spectral library starting from a proteome in fasta format.

The script runs through the following steps:
- In silico cleavage of proteins from the fasta file
- Remove peptide redundancy
- Add all variations of variable modifications (max 7 PTMs/peptide)
- Add variations on charge state
- Predict spectra with MS2PIP
- Write to various output file formats


Unspecific cleavage (e.g. for immunopeptidomics) is supported by setting
``cleavage_rule`` to ``unspecific``.


Decoys added by reversing sequences, keeping the N-terminal residue inplace.


Modifications:
- Peptides can carry only one modification per site (side chain or terminus).
- Protein terminal modifications take precedence over peptide terminal modifications.
- Terminal modifications can have site specificity (e.g. N-term K or N-term P).

"""
from __future__ import annotations

__author__ = "Ralf Gabriels"
__copyright__ = "CompOmics"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__email__ = "Ralf.Gabriels@ugent.be"

import argparse
import json
import logging
import multiprocessing
import multiprocessing.dummy
from collections import defaultdict
from functools import cmp_to_key, partial
from itertools import chain, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from ms2pip.ms2pip_tools import spectrum_output
from ms2pip.ms2pipC import MODELS, MS2PIP
from ms2pip.peptides import Modifications as MS2PIPModifications
from ms2pip.retention_time import RetentionTime
from pydantic import BaseModel, validator
from pyteomics.fasta import FASTA, Protein, decoy_db
from pyteomics.parser import icleave
from rich.logging import RichHandler
from rich.progress import track

logger = logging.getLogger(__name__)


class Peptide(BaseModel):
    """Peptide representation within the fasta2speclib search space."""
    sequence: str
    proteins: List[str]
    is_n_term: Optional[bool] = None
    is_c_term: Optional[bool] = None
    modification_options: List[str] = None
    charge_options: List[int] = None


class ModificationConfig(BaseModel):
    """Configuration for a single modification in the search space."""
    name: str
    mass_shift: float
    unimod_accession: Optional[int] = None
    amino_acid: Optional[str] = None
    peptide_n_term: Optional[bool] = False
    protein_n_term: Optional[bool] = False
    peptide_c_term: Optional[bool] = False
    protein_c_term: Optional[bool] = False
    fixed: Optional[bool] = False

    @validator("protein_c_term", always=True)  # Validate on last target in model
    def modification_must_have_target(cls, v, values):
        target_fields = [
            "amino_acid",
            "peptide_n_term",
            "protein_n_term",
            "peptide_c_term",
            "protein_c_term",
        ]
        if not any(t in values and values[t] for t in target_fields):
            raise ValueError(
                "Modifications must have at least one target (amino acid or N/C-term)."
            )
        return v


DEFAULT_MODIFICATIONS = [
    ModificationConfig(
        name="Oxidation",
        unimod_accession=35,
        mass_shift=15.994915,
        amino_acid="M",
    ),
    ModificationConfig(
        name="Carbamidomethyl",
        mass_shift=57.021464,
        unimod_accession=4,
        amino_acid="C",
        fixed=True,
    ),
]


class Configuration(BaseModel):
    """Configuration for fasta2speclib."""

    fasta_filename: Union[str, Path]
    output_filename: Optional[str] = None
    output_filetype: Optional[List[str]] = None
    charges: List[int] = [2, 3]
    min_length: int = 8
    max_length: int = 30
    cleavage_rule: str = "trypsin"
    missed_cleavages: int = 2
    semi_specific: bool = False
    modifications: List[ModificationConfig] = DEFAULT_MODIFICATIONS
    max_variable_modifications: int = 3
    min_precursor_mz: Optional[float] = None
    max_precursor_mz: Optional[float] = None
    ms2pip_model: str = "HCD"
    add_decoys: float = False
    add_retention_time: float = True
    deeplc: dict = dict()
    batch_size: int = 10000
    num_cpu: Optional[int] = None

    @validator("output_filetype")
    def _validate_output_filetypes(cls, v):
        allowed_types = ["msp", "mgf", "bibliospec", "spectronaut", "dlib", "hdf"]
        v = [filetype.lower() for filetype in v]
        for filetype in v:
            if filetype not in allowed_types:
                raise ValueError(
                    f"File type `{filetype}` not recognized. Should be one of "
                    f"`{allowed_types}`."
                )
        return v

    @validator("modifications")
    def _validate_modifications(cls, v):
        if all(isinstance(m, ModificationConfig) for m in v):
            return v
        elif all(isinstance(m, dict) for m in v):
            return [ModificationConfig(**modification) for modification in v]
        else:
            raise ValueError(
                "Modifications should be a list of dicts or ModificationConfig objects."
            )

    @validator("ms2pip_model")
    def _validate_ms2pip_model(cls, v):
        if v not in MODELS.keys():
            raise ValueError(
                f"MS²PIP model `{v}` not recognized. Should be one of " f"`{MODELS.keys()}`."
            )
        return v

    @validator("num_cpu")
    def _validate_num_cpu(cls, v):
        available_cpus = multiprocessing.cpu_count()
        if not v or not 0 < v < available_cpus:
            return available_cpus
        else:
            return v

    def get_output_filename(self):
        if self.output_filename:
            return self.output_filename
        else:
            return str(Path(self.fasta_filename).with_suffix(""))


class Fasta2SpecLib:
    """Generate an MS²PIP- and DeepLC-predicted spectral library from a FASTA file."""

    def __init__(
        self,
        fasta_filename: Union[str, Path],
        output_filename: Optional[Union[str, Path]] = None,
        config: Optional[Union[Configuration, dict]] = None,
    ):
        """
        Generate an MS²PIP- and DeepLC-predicted spectral library.

        fasta_filename: str, Path
            Path to input FASTA file.
        output_filename: str, Path
            Stem for output filenames. For instance, ``./output`` would result in
            ``./output.msp``. If ``None``, the output filename will be based on the
            input FASTA filename.
        config: Configuration, dict, optional
            Configuration of fasta2speclib. See documentation for more info

        """
        # Parse configuration
        if config:
            if isinstance(config, dict):
                config["fasta_filename"] = fasta_filename
                config["output_filename"] = output_filename
                config = Configuration.parse_obj(config)
            elif isinstance(config, Configuration):
                config.fasta_filename = fasta_filename
                config.output_filename = output_filename
            else:
                raise TypeError(f"Invalid type for configuration: `{type(config)}`.")
        else:
            config = Configuration(fasta_filename=fasta_filename, output_filename=output_filename)

        # `unspecific` is not an option in pyteomics.parser.icleave, so we configure
        # the settings for unspecific cleavage here.
        if config.cleavage_rule == "unspecific":
            config.missed_cleavages = config.max_length
            config.cleavage_rule = r"(?<=[A-Z])"

        # Setup multiprocessing, using a dummy pool if num_cpu is 1
        if config.num_cpu != 1:
            self.Pool = multiprocessing.Pool
        else:
            self.Pool = multiprocessing.dummy.Pool

        self.config = config
        self.rt_predictor = self._get_rt_predictor(config)
        self.ms2pip_params = self._prepare_ms2pip_params(config)

    def run(self):
        """Run the library generation pipeline."""
        peptides = self.prepare_search_space()
        batches = self.peptides_to_batches(peptides, self.config.batch_size)

        # Run in batches to avoid memory issues
        for batch_id, batch_peptides in enumerate(batches):
            logger.info(f"Processing batch {batch_id + 1}/{len(batches)}...")
            self.process_batch(batch_id, batch_peptides)

    def prepare_search_space(self) -> List[Peptide]:
        """Prepare peptide search space from FASTA file."""
        logger.info("Preparing search space...")

        # Setup database, with decoy configuration if required
        n_proteins = count_fasta_entries(self.config.fasta_filename)
        if self.config.add_decoys:
            fasta_db = decoy_db(
                self.config.fasta_filename,
                mode="reverse",
                decoy_only=False,
                keep_nterm=True,
            )
        else:
            fasta_db = FASTA(self.config.fasta_filename)
            n_proteins *= 2

        # Read proteins and digest to peptides
        with self.Pool(self.config.num_cpu) as pool:
            partial_digest_protein = partial(
                self._digest_protein,
                min_length=self.config.min_length,
                max_length=self.config.max_length,
                cleavage_rule=self.config.cleavage_rule,
                missed_cleavages=self.config.missed_cleavages,
                semi_specific=self.config.semi_specific,
            )
            results = track(
                pool.imap(partial_digest_protein, fasta_db),
                total=n_proteins,
                description="Digesting proteins...",
                transient=True,
            )
            peptides = list(chain.from_iterable(results))

        # Remove redundancy in peptides and combine protein lists
        peptide_dict = dict()
        for peptide in track(
            peptides,
            description="Removing peptide redundancy...",
            transient=True,
        ):
            if peptide.sequence in peptide_dict:
                peptide_dict[peptide.sequence].proteins.extend(peptide.proteins)
            else:
                peptide_dict[peptide.sequence] = peptide
        peptides = list(peptide_dict.values())

        # Add modification and charge permutations
        modifications_by_target = self._get_modifications_by_target(self.config.modifications)
        modification_options = []
        with self.Pool(self.config.num_cpu) as pool:
            partial_get_modification_versions = partial(
                self._get_modification_versions,
                modifications=self.config.modifications,
                modifications_by_target=modifications_by_target,
                max_variable_modifications=self.config.max_variable_modifications,
            )
            modification_options = pool.imap(partial_get_modification_versions, peptides)
            for pep, mod_opt in track(
                zip(peptides, modification_options),
                description="Adding modifications...",
                total=len(peptides),
                transient=True,
            ):
                pep.modification_options = mod_opt
                pep.charge_options = self.config.charges

        logger.info(f"Search space contains {len(peptides)} peptides.")
        return peptides

    @staticmethod
    def peptides_to_batches(peptides: List[Peptide], batch_size: int) -> List[List[Peptide]]:
        """Divide peptides into batches for batch-based processing."""
        return [peptides[i : i + batch_size] for i in range(0, len(peptides), batch_size)]

    def process_batch(self, batch_id, batch_peptides):
        """Predict and write library for a batch of peptides."""
        # Generate MS²PIP input
        logger.info("Generating MS²PIP input...")
        peprec = self._peptides_to_peprec(batch_peptides)
        logger.info(f"Chunk contains {len(peprec)} peptidoforms.")

        # Filter on precursor m/z
        if self.config.min_precursor_mz and self.config.max_precursor_mz:
            mods = MS2PIPModifications()
            mods.add_from_ms2pip_modstrings(self.ms2pip_params["ms2pip"]["ptm"])
            precursor_mz = peprec.apply(
                lambda x: mods.calc_precursor_mz(x["peptide"], x["modifications"], x["charge"])[1],
                axis=1,
            )
            before = len(peprec)
            peprec = (
                peprec[
                    (self.config.min_precursor_mz <= precursor_mz)
                    & (precursor_mz <= self.config.max_precursor_mz)
                ]
                .reset_index(drop=True)
                .copy()
            )
            after = len(peprec)
            logger.info(f"Filtered batch on precursor m/z: {before} -> {after}")

        # Predict retention time
        if self.config.add_retention_time:
            logger.info("Predicting retention times with DeepLC...")
            self.rt_predictor.add_rt_predictions(peprec)

        # Predict spectra
        logger.info("Predicting spectra with MS²PIP...")
        ms2pip = MS2PIP(
            peprec,
            num_cpu=self.config.num_cpu,
            params=self.ms2pip_params,
            return_results=True,
            add_retention_time=False,
        )
        predictions = ms2pip.run()

        # Write output
        logger.info("Writing output...")
        self._write_predictions(
            predictions,
            peprec,
            self.config.output_filetype,
            self.config.get_output_filename(),
            self.ms2pip_params,
            append=batch_id != 0,
        )

    @staticmethod
    def _get_rt_predictor(config: Configuration) -> RetentionTime:
        """Initialize and return MS²PIP wrapper for DeepLC predictor."""
        if config.add_retention_time:
            logger.debug("Initializing DeepLC predictor")
            if not config.deeplc:
                config.deeplc = {"calibration_file": None}
            if not "n_jobs" in config.deeplc:
                config.deeplc["n_jobs"] = config.num_cpu
            rt_predictor = RetentionTime(config=config.dict())
        else:
            rt_predictor = None
        return rt_predictor

    @staticmethod
    def _prepare_ms2pip_params(config: Configuration) -> dict:
        """Prepare MS²PIP parameters from fasta2speclib configuration."""
        ms2pip_params = {
            "ms2pip": {
                "model": config.ms2pip_model,
                "frag_error": 0.02,
                "ptm": [
                    "{},{},opt,N-term".format(mod.name, mod.mass_shift)
                    if mod.peptide_n_term or mod.protein_n_term
                    else "{},{},opt,C-term".format(mod.name, mod.mass_shift)
                    if mod.peptide_c_term or mod.protein_c_term
                    else "{},{},opt,{}".format(mod.name, mod.mass_shift, mod.amino_acid)
                    for mod in config.modifications
                ],
                "sptm": [],
                "gptm": [],
            }
        }
        return ms2pip_params

    @staticmethod
    def _digest_protein(
        protein: Protein,
        min_length: int = 8,
        max_length: int = 30,
        cleavage_rule: str = "trypsin",
        missed_cleavages: int = 2,
        semi_specific: bool = False,
    ) -> List[Peptide]:
        """Digest protein sequence and return a list of validated peptides."""

        def valid_residues(sequence: str) -> bool:
            return not any(aa in sequence for aa in ["B", "J", "O", "U", "X", "Z"])

        def parse_peptide(
            start_position: int,
            sequence: str,
            protein: Protein,
        ) -> Peptide:
            """Parse result from parser.icleave into Peptide."""
            return Peptide(
                sequence=sequence,
                # Assumes protein ID is description until first space
                proteins=[protein.description.split(" ")[0]],
                is_n_term=start_position == 0,
                is_c_term=start_position + len(sequence) == len(protein.sequence),
            )

        peptides = [
            parse_peptide(start, seq, protein)
            for start, seq in icleave(
                protein.sequence,
                cleavage_rule,
                missed_cleavages=missed_cleavages,
                min_length=min_length,
                max_length=max_length,
                semi=semi_specific,
            )
            if valid_residues(seq)
        ]

        return peptides

    @staticmethod
    def _get_modifications_by_target(
        modifications,
    ) -> Dict[str, Dict[str, List[ModificationConfig]]]:
        """Restructure variable modifications to options per side chain or terminus."""
        modifications_by_target = {
            "sidechain": defaultdict(lambda: [None]),
            "peptide_n_term": defaultdict(lambda: [None]),
            "peptide_c_term": defaultdict(lambda: [None]),
            "protein_n_term": defaultdict(lambda: [None]),
            "protein_c_term": defaultdict(lambda: [None]),
        }

        def add_mod(mod, target, amino_acid):
            if amino_acid:
                modifications_by_target[target][amino_acid].append(mod)
            else:
                modifications_by_target[target]["any"].append(mod)

        for mod in modifications:
            if mod.fixed:
                continue
            if mod.peptide_n_term:
                add_mod(mod, "peptide_n_term", mod.amino_acid)
            elif mod.peptide_c_term:
                add_mod(mod, "peptide_c_term", mod.amino_acid)
            elif mod.protein_n_term:
                add_mod(mod, "protein_n_term", mod.amino_acid)
            elif mod.protein_c_term:
                add_mod(mod, "protein_c_term", mod.amino_acid)
            else:
                add_mod(mod, "sidechain", mod.amino_acid)

        return {k: dict(v) for k, v in modifications_by_target.items()}

    # TODO: Make adding modifications more efficient
    @staticmethod
    def _get_modification_versions(
        peptide: Peptide,
        modifications: List[ModificationConfig],
        modifications_by_target: Dict[str, Dict[str, List[ModificationConfig]]],
        max_variable_modifications: int = 3,
    ) -> List[str]:
        """Get MS²PIP modification strings for all potential versions."""
        possibilities_by_site = defaultdict(list)

        # Generate dictionary of positions per amino acid
        pos_dict = defaultdict(list)
        for pos, aa in enumerate(peptide.sequence):
            pos_dict[aa].append(pos + 1)
        # Map modifications to positions
        for aa in set(pos_dict).intersection(set(modifications_by_target["sidechain"])):
            possibilities_by_site.update(
                {pos: modifications_by_target["sidechain"][aa] for pos in pos_dict[aa]}
            )

        # Assign possible modifications per terminus
        for terminus, position, specificity in [
            ("peptide_n_term", 0, None),
            ("peptide_c_term", -1, None),
            ("protein_n_term", 0, "is_n_term"),
            ("protein_c_term", -1, "is_c_term"),
        ]:
            if specificity is None or getattr(peptide, specificity):
                for site, mods in modifications_by_target[terminus].items():
                    if site == "any" or peptide.sequence[position] == site:
                        possibilities_by_site[position].extend(mods)

        # Override with fixed modifications
        for mod in modifications:
            aa = mod.amino_acid
            # Skip variable modifications
            if not mod.fixed:
                continue
            # Assign if specific aa matches or if no aa is specified for each terminus
            for terminus, position, specificity in [
                ("peptide_n_term", 0, None),
                ("peptide_c_term", -1, None),
                ("protein_n_term", 0, "is_n_term"),
                ("protein_c_term", -1, "is_c_term"),
            ]:
                if getattr(mod, terminus):  # Mod has this terminus
                    if specificity is None or getattr(peptide, specificity):  # Specificity matches
                        if not aa or (aa and peptide.sequence[position] == aa):  # Aa matches
                            possibilities_by_site[position] = [mod]  # Override with fixed mod
                    break  # Allow `else: if amino_acid` if no terminus matches
            # Assign if fixed modification is not terminal and specific aa matches
            else:
                if aa:
                    for pos in pos_dict[aa]:
                        possibilities_by_site[pos] = [mod]

        # Get all possible combinations of modifications for all sites
        mod_permutations = product(*possibilities_by_site.values())
        mod_positions = possibilities_by_site.keys()

        # Filter by max modified sites (avoiding combinatorial explosion)
        mod_permutations = filter(
            lambda mods: sum([1 for m in mods if m is not None and not m.fixed])
            <= max_variable_modifications,
            mod_permutations,
        )

        def _compare_minus_one_larger(a, b):
            """Custom comparison function where `-1` is always larger."""
            if a[0] == -1:
                return 1
            elif b[0] == -1:
                return -1
            else:
                return a[0] - b[0]

        # Get MS²PIP modifications strings for each combination
        mod_strings = []
        for p in mod_permutations:
            if p == [""]:
                mod_strings.append("-")
            else:
                mods = sorted(zip(mod_positions, p), key=cmp_to_key(_compare_minus_one_larger))
                mod_strings.append("|".join(f"{p}|{m.name}" for p, m in mods if m))

        return mod_strings

    @staticmethod
    def _peptides_to_peprec(peptides: List[Peptide]) -> pd.DataFrame:
        """Convert a list of peptides to a PeptideRecord DataFrame."""
        peprec = pd.DataFrame(
            [
                {
                    "peptide": peptide.sequence,
                    "modifications": modifications,
                    "charge": charge,
                    "protein_list": peptide.proteins
                }
                for peptide in peptides
                for charge in peptide.charge_options
                for modifications in peptide.modification_options
            ],
            columns=["spec_id", "peptide", "modifications", "charge", "protein_list"],
        )
        peprec["spec_id"] = peprec.index
        return peprec

    @staticmethod
    def _write_predictions(
        predictions: pd.DataFrame,
        peprec: pd.DataFrame,
        filetypes: List[str],
        filename: str,
        ms2pip_params: Dict,
        append: bool = False,
    ):
        """Write predictions (for batch) to requested output file formats."""
        write_mode = "a" if append else "w"
        if "hdf" in filetypes:
            logger.info(f"Writing results to {filename}_predictions.hdf")
            predictions.astype(str).to_hdf(
                f"{filename}_predictions.hdf",
                key="table",
                format="table",
                complevel=3,
                complib="zlib",
                mode=write_mode,
                append=append,
                min_itemsize=50,
            )
        spec_out = spectrum_output.SpectrumOutput(
            predictions,
            peprec,
            ms2pip_params["ms2pip"],
            output_filename=filename,
            write_mode=write_mode,
        )
        if "msp" in filetypes:
            spec_out.write_msp()
        if "mgf" in filetypes:
            spec_out.write_mgf()
        if "bibliospec" in filetypes:
            spec_out.write_bibliospec()
        if "spectronaut" in filetypes:
            spec_out.write_spectronaut()
        if "dlib" in filetypes:
            spec_out.write_dlib()


def count_fasta_entries(filename: Union[str, Path]) -> int:
    """Count the number of entries in a FASTA file."""
    with open(filename, "rt") as f:
        count = 0
        for line in f:
            if line[0] == ">":
                count += 1
    return count


def _argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Create an MS2PIP- and DeepLC-predicted spectral library, starting from a "
            "FASTA file."
        )
    )
    parser.add_argument(
        "fasta_filename",
        action="store",
        help="Path to the FASTA file containing protein sequences",
    )
    parser.add_argument(
        "-o",
        dest="output_filename",
        action="store",
        help="Name for output file(s) (if not given, derived from FASTA file)",
    )
    parser.add_argument(
        "-c",
        dest="config_filename",
        action="store",
        help="Name of configuration json file (default: fasta2speclib_config.json)",
    )

    args = parser.parse_args()
    return args


def main():
    """Command line entrypoint for fasta2speclib."""
    # Configure logging
    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[RichHandler(rich_tracebacks=True, show_level=True, show_path=False)],
    )
    logging.getLogger("ms2pip").setLevel(logging.WARNING)
    logging.getLogger("deeplc").setLevel(logging.WARNING)

    # Get configuration from CLI and config file
    args = _argument_parser()
    with open(args.config_filename, "rt") as config_file:
        config_dict = json.load(config_file)

    # Run fasta2speclib
    logger.info("Starting library generation pipeline...")
    f2sl = Fasta2SpecLib(args.fasta_filename, args.output_filename, config_dict)
    f2sl.run()
    logger.info("Done!")


if __name__ == "__main__":
    main()
