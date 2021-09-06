import ms2pip.peptides


class TestModifications:
    def test_add_from_ms2pip_modstrings(self):
        mods = ms2pip.peptides.Modifications()
        mods.add_from_ms2pip_modstrings([
            "Oxidation,15.994915,opt,M",
            "Acetyl,42.010565,opt,N-term",
            "Methyl,14.01565,opt,L",
        ])

        assert mods.modifications['ptm']["Oxidation"]["amino_acid"] == "M"
        assert mods.modifications['ptm']["Acetyl"]["mass_shift"] == 42.010565
        assert mods.modifications['ptm']["Methyl"]["mass_shift"] == 14.01565

    def test_get_mass_shifts(self):
        mods = ms2pip.peptides.Modifications()

        mods.add_from_ms2pip_modstrings([
            "Oxidation,15.994915,opt,M"
        ])
        assert mods.mass_shifts["Oxidation"] == 15.994915

        # Test cache clear after adding new modifications
        mods.add_from_ms2pip_modstrings([
            "Acetyl,42.010565,opt,N-term",
        ])
        assert mods.mass_shifts["Acetyl"] == 42.010565
