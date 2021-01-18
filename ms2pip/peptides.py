"""
MS2PIP / PEPREC modification handling
"""

import itertools
import tempfile

from pyteomics import mass

AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

AMINO_ACID_MASSES = [
    71.037114,
    103.00919,
    115.026943,
    129.042593,
    147.068414,
    57.021464,
    137.058912,
    113.084064,
    128.094963,
    131.040485,
    114.042927,
    97.052764,
    128.058578,
    156.101111,
    87.032028,
    101.047679,
    99.068414,
    186.079313,
    163.063329,
    # 147.0354  # iTRAQ fixed N-term modification (gets written to amino acid masses file)
]

AMINO_ACID_IDS = {a: i for i, a in enumerate(AMINO_ACIDS)}

PROTON_MASS = 1.007825032070059


class Modifications:
    def __init__(self):
        """
        MS2PIP / PEPREC modification handling
        """
        self.modifications = {}
        self._mass_shifts = None
        self._ptm_ids = None
        self._next_mod_id = 38  # Omega compatibility (mutations)

    def add_from_ms2pip_modstrings(self, modstrings, mod_type='ptm'):
        """
        Add modifications from MS2PIP modstring list

        Parameters
        ----------
        modstrings: list(str)
            List of MS2PIP modstrings

        Example
        -------
        >>> ms2pip_ptms = [
        ...     "Oxidation,15.994915,opt,M",
        ...     "Acetyl,42.010565,opt,N-term",
        ... ]
        ... mods = Modifications()
        ... mods.add_from_ms2pip_modstrings(ms2pip_ptms)
        """

        if mod_type not in self.modifications:
            self.modifications[mod_type] = {}

        # TODO: check opt
        for mod in modstrings:
            mod_name, mass_shift, opt, amino_acid = mod.split(",")

            if amino_acid == "N-term":
                amino_acid_id = -1
            elif amino_acid == "C-term":
                amino_acid_id = -2
            elif amino_acid in AMINO_ACID_IDS:
                amino_acid_id = AMINO_ACID_IDS[amino_acid]
            else:
                # TODO: raise or at least log error?
                continue

            self.modifications[mod_type][mod_name] = {
                "mass_shift": float(mass_shift),
                "amino_acid": amino_acid,
                "amino_acid_id": amino_acid_id,
                "mod_id": self._next_mod_id
            }
            self._next_mod_id += 1

        self._mass_shifts = None

    @property
    def _all_modifications(self):
        return itertools.chain.from_iterable(
            (mods.items() for mods in self.modifications.values())
        )

    @property
    def mass_shifts(self):
        """
        Return modification name -> mass shift mapping.
        """
        if not self._mass_shifts:
            self._mass_shifts = {name: mod["mass_shift"]
                                 for name, mod in self._all_modifications}
        return self._mass_shifts

    @property
    def ptm_ids(self):
        """
        Return modification name -> modification id mapping.
        """
        if not self._ptm_ids:
            self._ptm_ids = {name: mod["mod_id"]
                             for name, mod in self._all_modifications}
        return self._ptm_ids

    def write_modifications_file(self, mod_type='ptm'):
        mod_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="\n")
        mod_file.write("{}\n".format(len(self.modifications[mod_type])))
        for name, mod in self.modifications[mod_type].items():
            mod_file.write(
                "{},1,{},{}\n".format(mod["mass_shift"], mod['amino_acid_id'], mod['mod_id'])
            )
        mod_file.close()
        return mod_file.name

    def calc_precursor_mz(self, peptide, modifications, charge):
        """
        Calculate precursor mass and mz for given peptide and modification list,
        taking the modifications into account.

        Note: This method does not use the build-in Pyteomics modification handling, as
        that would require a known atomic composition of the modification.

        Parameters
        ----------
        peptide: str
            stripped peptide sequence

        modifications: str
            MS2PIP-style formatted modifications list (e.g. `0|Acetyl|2|Oxidation`)

        charge: int
            precursor charge

        Returns
        -------
        prec_mass, prec_mz: tuple(float, float)
        """

        charge = int(charge)
        unmodified_mass = mass.fast_mass(peptide)
        mods_massses = sum(
            [self.mass_shifts[mod] for mod in modifications.split("|")[1::2]]
        )
        prec_mass = unmodified_mass + mods_massses
        prec_mz = (prec_mass + charge * PROTON_MASS) / charge
        return prec_mass, prec_mz


def write_amino_accid_masses():
    # Create amino acid MASSES file
    # to be compatible with Omega
    # that might have fixed modifications
    amino_file = tempfile.NamedTemporaryFile(delete=False, mode="w", newline="\n")
    for m in AMINO_ACID_MASSES:
        amino_file.write("{}\n".format(m))
    amino_file.write("0\n")
    amino_file.close()
    return amino_file.name
