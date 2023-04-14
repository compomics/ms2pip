class MS2PIPError(Exception):
    pass


class InvalidPeptidoformError(MS2PIPError):
    pass


class InvalidInputError(MS2PIPError):
    pass


class UnresolvableModificationError(MS2PIPError):
    pass


class UnknownOutputFormatError(ValueError):
    pass


class UnknownModelError(ValueError):
    pass


class MissingConfigurationError(MS2PIPError):
    pass


class InvalidAminoAcidError(MS2PIPError):
    pass


class UnsupportedSpectrumFiletypeError(MS2PIPError):
    pass


class InvalidSpectrumError(MS2PIPError):
    pass


class NoMatchingSpectraFound(MS2PIPError):
    pass


class TitlePatternError(MS2PIPError):
    pass


class InvalidXGBoostModelError(MS2PIPError):
    pass
