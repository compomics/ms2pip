import tempfile
from pathlib import Path

import numpy as np
import pytest
from psm_utils import PSM, Peptidoform

from ms2pip.result import ProcessingResult
from ms2pip.spectrum_output import DLIB, MSP, Bibliospec


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

    def test_dlib_database_creation(self):
        """Test that DLIB file creation works with SQLAlchemy (integration test)."""
        # Create test data
        pep = Peptidoform("ACDE/2")
        psm = PSM(
            peptidoform=pep,
            spectrum_id=1,
            retention_time=100.0,
            protein_list=["PROT1", "PROT2"],
        )
        result = ProcessingResult(
            psm_index=0,
            psm=psm,
            theoretical_mz={
                "b": np.array([72.04435, 175.05354, 290.08047], dtype=np.float32),
                "y": np.array([148.0604, 263.0873, 366.0965], dtype=np.float32),
            },
            predicted_intensity={
                "b": np.array([0.1, 0.5, 0.3], dtype=np.float32),
                "y": np.array([0.8, 0.6, 0.2], dtype=np.float32),
            },
            observed_intensity=None,
            correlation=None,
            feature_vectors=None,
        )

        # Write DLIB file
        with tempfile.TemporaryDirectory() as tmpdir:
            dlib_file = Path(tmpdir) / "test.dlib"
            with DLIB(dlib_file) as writer:
                writer.write([result])

            # Verify file was created
            assert dlib_file.exists()

            # Verify database structure and content using SQLAlchemy
            from ms2pip._utils import dlib as dlib_module

            connection = dlib_module.open_sqlite(dlib_file)
            try:
                # Test that metadata table exists and has version
                from sqlalchemy import select

                version = connection.execute(
                    select(dlib_module.Metadata.c.Value).where(
                        dlib_module.Metadata.c.Key == "version"
                    )
                ).scalar()
                assert version == dlib_module.DLIB_VERSION

                # Test that Entry table has data (select specific columns to avoid nullable CompressedArray)
                from sqlalchemy import func

                entry_count = connection.execute(
                    select(func.count()).select_from(dlib_module.Entry)
                ).scalar()
                assert entry_count == 1

                # Select specific non-nullable columns
                entry = connection.execute(
                    select(
                        dlib_module.Entry.c.PeptideSeq,
                        dlib_module.Entry.c.PrecursorCharge,
                        dlib_module.Entry.c.RTInSeconds,
                        dlib_module.Entry.c.MassArray,
                        dlib_module.Entry.c.IntensityArray,
                    )
                ).fetchone()
                assert entry.PeptideSeq == "ACDE"
                assert entry.PrecursorCharge == 2
                assert entry.RTInSeconds == 100.0
                assert len(entry.MassArray) == 6  # 3 b-ions + 3 y-ions
                assert len(entry.IntensityArray) == 6

                # Test that PeptideToProtein table has data
                peptide_to_proteins = connection.execute(
                    select(dlib_module.PeptideToProtein)
                ).fetchall()
                assert len(peptide_to_proteins) == 2
                proteins = {p.ProteinAccession for p in peptide_to_proteins}
                assert proteins == {"PROT1", "PROT2"}
                assert all(p.PeptideSeq == "ACDE" for p in peptide_to_proteins)
            finally:
                connection.close()

    def test_dlib_multiple_results(self):
        """Test writing multiple ProcessingResults to DLIB file."""
        # Create multiple test results
        results = []
        for i, seq in enumerate(["ACDE/2", "PEPTIDE/2", "TESTK/2"]):
            pep = Peptidoform(seq)
            psm = PSM(
                peptidoform=pep,
                spectrum_id=i,
                retention_time=100.0 + i * 10,
                protein_list=[f"PROT{i}"],
            )
            result = ProcessingResult(
                psm_index=i,
                psm=psm,
                theoretical_mz={
                    "b": np.array([72.04435, 175.05354], dtype=np.float32),
                    "y": np.array([148.0604, 263.0873], dtype=np.float32),
                },
                predicted_intensity={
                    "b": np.array([0.1, 0.5], dtype=np.float32),
                    "y": np.array([0.8, 0.6], dtype=np.float32),
                },
                observed_intensity=None,
                correlation=None,
                feature_vectors=None,
            )
            results.append(result)

        # Write DLIB file
        with tempfile.TemporaryDirectory() as tmpdir:
            dlib_file = Path(tmpdir) / "test_multiple.dlib"
            with DLIB(dlib_file) as writer:
                writer.write(results)

            # Verify all entries were written
            from sqlalchemy import select

            from ms2pip._utils import dlib as dlib_module

            connection = dlib_module.open_sqlite(dlib_file)
            try:
                # Select specific columns to avoid nullable CompressedArray
                entries = connection.execute(
                    select(
                        dlib_module.Entry.c.PeptideSeq,
                        dlib_module.Entry.c.RTInSeconds,
                    )
                ).fetchall()
                assert len(entries) == 3

                peptides = {e.PeptideSeq for e in entries}
                assert peptides == {"ACDE", "PEPTIDE", "TESTK"}

                # Verify retention times
                rt_values = {e.RTInSeconds for e in entries}
                assert rt_values == {100.0, 110.0, 120.0}
            finally:
                connection.close()

    def test_dlib_sqlalchemy_select_syntax(self):
        """Test that SQLAlchemy v2 select() syntax works correctly."""
        # This test specifically verifies the SQLAlchemy v2 compatibility changes
        pep = Peptidoform("ACDE/2")
        psm = PSM(
            peptidoform=pep,
            spectrum_id=1,
            retention_time=100.0,
            protein_list=["PROT1"],
        )
        result = ProcessingResult(
            psm_index=0,
            psm=psm,
            theoretical_mz={
                "b": np.array([72.04435], dtype=np.float32),
                "y": np.array([148.0604], dtype=np.float32),
            },
            predicted_intensity={
                "b": np.array([0.5], dtype=np.float32),
                "y": np.array([0.8], dtype=np.float32),
            },
            observed_intensity=None,
            correlation=None,
            feature_vectors=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dlib_file = Path(tmpdir) / "test_sqlalchemy.dlib"
            with DLIB(dlib_file) as writer:
                writer.write([result])

            # Test the specific SQLAlchemy operations that were modified
            from sqlalchemy import select

            from ms2pip._utils import dlib as dlib_module

            connection = dlib_module.open_sqlite(dlib_file)
            try:
                # Test select(Table) syntax (changed from Table.select())
                peptide_to_protein_results = connection.execute(
                    select(dlib_module.PeptideToProtein).where(
                        dlib_module.PeptideToProtein.c.ProteinAccession == "PROT1"
                    )
                ).fetchall()
                assert len(peptide_to_protein_results) == 1
                assert peptide_to_protein_results[0].PeptideSeq == "ACDE"

                # Test select(column) syntax (changed from select([column]))
                version = connection.execute(
                    select(dlib_module.Metadata.c.Value).where(
                        dlib_module.Metadata.c.Key == "version"
                    )
                ).scalar()
                assert version is not None
                assert version == dlib_module.DLIB_VERSION
            finally:
                connection.close()

    def test_dlib_missing_retention_time(self):
        """Test that DLIB writing raises error when retention time is missing."""
        pep = Peptidoform("ACDE/2")
        psm = PSM(peptidoform=pep, spectrum_id=1)  # No retention_time
        result = ProcessingResult(
            psm_index=0,
            psm=psm,
            theoretical_mz={"b": np.array([72.04435], dtype=np.float32)},
            predicted_intensity={"b": np.array([0.5], dtype=np.float32)},
            observed_intensity=None,
            correlation=None,
            feature_vectors=None,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            dlib_file = Path(tmpdir) / "test_no_rt.dlib"
            with pytest.raises(ValueError, match="Retention time required"):
                with DLIB(dlib_file) as writer:
                    writer.write([result])
