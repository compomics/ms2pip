import unittest
import numpy as np
import ms2pip.match_spectra
from operator import itemgetter


class TestMatchSpectra(unittest.TestCase):
    def test_get_intense_mzs(self):
        mzs = np.array([72.04435, 143.08147, 214.11859, 285.1557, 414.19827, 527.28235, 598.31946, 697.3879, 147.11276, 246.18117, 317.21826, 430.30234, 559.345, 630.3821, 701.4192, 772.4563], dtype=np.float32)
        intensities = np.array([0.000340063, 0.186675, 0.0165939, 0.0138825, 0, 0, 0, 0, 0.0670459, 0.147733, 0.046132, 0.00865729, 0.181739, 0.161184, 0.15575, 0.0142658], dtype=np.float32)
        top5 = [np.float32(x) for x in (143.08147, 559.345, 630.3821, 701.4192, 246.18117)]

        self.assertEqual(top5[:3], ms2pip.match_spectra.get_intense_mzs(mzs, intensities))
        self.assertEqual([], ms2pip.match_spectra.get_intense_mzs(mzs, intensities, n=0))
        self.assertEqual(top5, ms2pip.match_spectra.get_intense_mzs(mzs, intensities, n=5))
        self.assertEqual([x[0] for x in sorted(zip(mzs, intensities), key=itemgetter(1), reverse=True)], ms2pip.match_spectra.get_intense_mzs(mzs, intensities, n=len(mzs)))

    def test_match_mzs(self):
        mzs = np.array([72.04435, 143.08147, 214.11859, 285.1557, 414.19827, 527.28235, 598.31946, 697.3879, 147.11276, 246.18117, 317.21826, 430.30234, 559.345, 630.3821, 701.4192, 772.4563], dtype=np.float32)
        top3 = [np.float32(x) for x in (143.08147, 559.345, 630.3821)]

        self.assertTrue(ms2pip.match_spectra.match_mzs(mzs, []))
        self.assertFalse(ms2pip.match_spectra.match_mzs([], []))
        self.assertFalse(ms2pip.match_spectra.match_mzs([], [3]))
        self.assertTrue(ms2pip.match_spectra.match_mzs(mzs, top3))
        self.assertFalse(ms2pip.match_spectra.match_mzs(mzs, [x + 0.02 for x in top3]))
        self.assertFalse(ms2pip.match_spectra.match_mzs(mzs, [x - 0.02 for x in top3]))
        self.assertTrue(ms2pip.match_spectra.match_mzs(mzs, [x + 0.02 for x in top3], max_error=0.05))
        self.assertTrue(ms2pip.match_spectra.match_mzs(mzs, [x - 0.02 for x in top3], max_error=0.05))
