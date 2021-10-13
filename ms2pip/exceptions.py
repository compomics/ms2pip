class MS2PIPError(Exception):
    pass


class InvalidPeptideError(MS2PIPError):
    pass


class UnknownModificationError(ValueError):
    pass


class InvalidPEPRECError(Exception):
    pass


class NoValidPeptideSequencesError(Exception):
    pass


class UnknownOutputFormatError(ValueError):
    pass


class UnknownFragmentationMethodError(ValueError):
    pass


class MissingConfigurationError(Exception):
    pass


class FragmentationModelRequiredError(Exception):
    pass


class InvalidModificationFormattingError(Exception):
    pass


class InvalidAminoAcidError(Exception):
    pass


class EmptySpectrumError(Exception):
    pass


class InvalidXGBoostModelError(Exception):
    pass
