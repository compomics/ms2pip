import bisect
import logging
from operator import itemgetter

import numpy as np
import pyteomics.mgf

logger = logging.getLogger("ms2pip.match_spectra")


def get_intense_mzs(mzs, intensity, n=3):
    return [x[0] for x in sorted(zip(mzs, intensity), key=itemgetter(1), reverse=True)[:n]]


def match_mzs(mzs, predicted, max_error=0.02):
    current = 0
    for pred in predicted:
        current = bisect.bisect_right(mzs, pred - max_error, lo=current)
        if current >= len(mzs) or mzs[current] >= pred + max_error:
            return False
    return current < len(mzs)


def get_predicted_peaks(pepids, mzs, intensities):
    # NOTE: we need to concatenate the predictions for b and y ions
    return dict(zip(pepids,
                    (sorted(get_intense_mzs(np.concatenate(_mzs, axis=None),
                                            np.concatenate(_intensities, axis=None)))
                     for _mzs, _intensities in zip(mzs, intensities))))


class MatchSpectra:
    DATA_COLS = ['spec_id', 'peptide', 'modifications', 'charge']

    def __init__(self, peprec, mods, pepids, predicted_mzs, predicted_intensities):
        """
        Initialise spectra matcher

        Parameters:
        -----------
        peprec: pandas.DataFrame
            PEPREC formatted input peptides
        mods: ms2pip.peptides.Modifications
            Modifications used in PEPREC file
        pepids: Iterable[str]
            Iterable of spec_id's from peprec input ordered matching the predictions
        predicted_mzs: Iterable[list[numpy.array[float32]]]
            Iterable of predicted m/z values
        predicted_mzs: Iterable[list[numpy.array[float32]]]
            Iterable of predicted intensities
        """
        self.peprec = peprec
        self.mods = mods
        self.predictions = get_predicted_peaks(pepids, predicted_mzs, predicted_intensities)
        self._generate_peptide_list()

    def _generate_peptide_list(self):
        peptides = [
            (
                spec_id,
                self.mods.calc_precursor_mz(peptide,
                                            modifications,
                                            charge)[1],
                self.predictions[spec_id]
            ) for spec_id, peptide, modifications, charge in self.peprec[self.DATA_COLS].values
        ]
        peptides.sort(key=itemgetter(1))
        self.peptides = peptides

    def match_mgfs(self, mgf_files, max_error=0.02):
        """
        Match predicted spectra to spectra in given MGF files.

        Paramters:
        ----------
        mgf_files: list[str]
            List of filenames of MGF files
        max_error: float, optional
            Maximum error for masses
        """
        logger.info("match predicted spectra to spectra in mgf files (%s)", mgf_files)
        precursors = [x[1] for x in self.peptides]

        for mgf_file in mgf_files:
            logger.debug("open %s", mgf_file)
            with pyteomics.mgf.read(mgf_file, use_header=False, convert_arrays=0, read_charges=False) as reader:
                for spectrum in reader:
                    if 'pepmass' not in spectrum['params']:
                        continue
                    pepmass = spectrum['params']['pepmass'][0]

                    # compare all peptides with a similar precursor m/z
                    i = bisect.bisect_right(precursors, pepmass - max_error)
                    while i < len(precursors) and precursors[i] < pepmass + max_error:
                        spec_id, _, pred_peaks = self.peptides[i]
                        if match_mzs(sorted(spectrum['m/z array']), pred_peaks, max_error=max_error):
                            yield spec_id, mgf_file, spectrum
                        i += 1

    def match_sqldb(self, sqldb_uri="postgresql:///ms2pip", max_error=0.02):
        """
        Match predicted spectra to given database of observed spectra.

        Paramters:
        ----------
        sqldb_uri: str
            URI of database to connect to
        max_error: float, optional
            Maximum error for masses
        """
        from ms2pip.sqldb import tables
        from sqlalchemy import select

        engine = tables.create_engine(sqldb_uri)
        precursors = np.fromiter((x[1] for x in self.peptides), dtype=np.float, count=len(self.peprec))
        gaps = np.where(np.diff(precursors) >= max_error)[0]

        with engine.connect() as connection:
            start = 0
            for end in gaps:
                for spec in connection.execute(
                        select([tables.spec, tables.specfile.c.filename])
                        .select_from(tables.spec.join(tables.specfile)).where(
                            tables.spec.c.pepmass > self.peptides[start][1] - max_error
                        ).where(
                            tables.spec.c.pepmass < self.peptides[end][1] + max_error
                        ).order_by(tables.spec.c.pepmass)):
                    for spec_id, mz, pred_peaks in self.peptides[start:end+1]:
                        if mz > spec.pepmass + max_error:
                            break
                        if mz < spec.pepmass - max_error:
                            start += 1
                            continue
                        if match_mzs(spec.mzs, pred_peaks, max_error=max_error):
                            yield (spec_id,
                                   spec.filename,
                                   {'params': {'title': spec.spec_id},
                                    'm/z array': spec.mzs})
                start = end + 1
