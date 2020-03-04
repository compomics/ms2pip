import pytest

import ms2pip.modifications


class TestModifications:
    def test_add_from_ms2pip_modstrings(self):
        mods = ms2pip.modifications.Modifications()
        mods.add_from_ms2pip_modstrings([
            "Oxidation,15.994915,opt,M",
            "Acetyl,42.010565,opt,N-term",
        ])

        assert mods.modifications["Oxidation"]["amino_acid"] == "M"
        assert mods.modifications["Acetyl"]["mass_shift"] == 42.010565

    def test_get_mass_shifts(self):
        mods = ms2pip.modifications.Modifications()

        mods.add_from_ms2pip_modstrings([
            "Oxidation,15.994915,opt,M"
        ])
        assert mods.get_mass_shifts()["Oxidation"] == 15.994915

        # Test cache clear after adding new modifications
        mods.add_from_ms2pip_modstrings([
            "Acetyl,42.010565,opt,N-term",
        ])
        assert mods.get_mass_shifts()["Acetyl"] == 42.010565
