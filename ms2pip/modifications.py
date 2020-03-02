"""
MS2PIP / PEPREC modification handling
"""

import functools


class Modifications:
    def __init__(self):
        """
        MS2PIP / PEPREC modification handling
        """
        self.modifications = dict()

    def add_from_ms2pip_modstrings(self, modstrings):
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

        for mod in modstrings:
            mod = mod.split(",")
            self.modifications[mod[0]] = {
                "mass_shift": float(mod[1]),
                "amino_acid": mod[3],
            }

        self.get_mass_shifts.cache_clear()

    @functools.lru_cache()
    def get_mass_shifts(self):
        """
        Return modification name -> mass shift mapping.
        """
        mass_shifts = {name: mod["mass_shift"] for name, mod in self.modifications.items()}
        return mass_shifts





