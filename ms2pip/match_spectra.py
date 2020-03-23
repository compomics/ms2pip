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
        if current >= len(mzs) or mzs[current] > pred + max_error:
            return False
    return current < len(mzs)


def get_predicted_peaks(pepids, mzs, intensities):
    return dict(zip(pepids,
                    (sorted(get_intense_mzs(np.concatenate(_mzs, axis=None),
                                            np.concatenate(_intensities, axis=None)))
                     for _mzs, _intensities in zip(mzs, intensities))))


class MatchSpectra:
    DATA_COLS = ['spec_id', 'peptide', 'modifications', 'charge']

    def __init__(self, peprec, mods, pepids, predicted_mzs, predicted_intensities):
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
                            yield spec_id, spectrum
                        i += 1
