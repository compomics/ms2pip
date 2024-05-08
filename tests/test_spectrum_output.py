from psm_utils import Peptidoform

from ms2pip.spectrum_output import MSP, Bibliospec, DLIB


class TestMSP:
    def test__format_modification_string(self):
        test_cases = [
            ("ACDE/2", "Mods=0"),
            ("AC[Carbamidomethyl]DE/2", "Mods=1/2,C,Carbamidomethyl"),
            ("[Glu->pyro-Glu]-EPEPTIDEK/2", "Mods=1/0,E,Glu->pyro-Glu"),
            ("PEPTIDEK-[Amidated]/2", "Mods=1/-1,K,Amidated"),
            ("AM[Oxidation]C[Carbamidomethyl]DE/2", "Mods=2/2,M,Oxidation/3,C,Carbamidomethyl"),
        ]

        for peptidoform_str, expected_output in test_cases:
            peptidoform = Peptidoform(peptidoform_str)
            assert MSP._format_modifications(peptidoform) == expected_output


class TestBiblioSpec:
    def test__format_modified_sequence(self):
        test_cases = [
            ("ACDE/2", "ACDE"),
            ("AC[Carbamidomethyl]DE/2", "AC[+57.0]DE"),
            ("[Glu->pyro-Glu]-EPEPTIDEK/2", "E[-18.0]PEPTIDEK"),
            ("PEPTIDEK-[Amidated]/2", "PEPTIDEK[-1.0]"),
            ("AM[Oxidation]C[Carbamidomethyl]DE/2", "AM[+16.0]C[+57.0]DE"),
        ]

        for peptidoform_str, expected_output in test_cases:
            peptidoform = Peptidoform(peptidoform_str)
            assert Bibliospec._format_modified_sequence(peptidoform) == expected_output


class TestDLIB:
    def test__format_modified_sequence(self):
        test_cases = [
            ("ACDE/2", "ACDE"),
            ("AC[Carbamidomethyl]DE/2", "AC[+57.021464]DE"),
            ("[Glu->pyro-Glu]-EPEPTIDEK/2", "E[-18.010565]PEPTIDEK"),
            ("PEPTIDEK-[Amidated]/2", "PEPTIDEK[-0.984016]"),
            ("AM[Oxidation]C[Carbamidomethyl]DE/2", "AM[+15.994915]C[+57.021464]DE"),
        ]

        for test_in, expected_out in test_cases:
            assert DLIB._format_modified_sequence(Peptidoform(test_in)) == expected_out
