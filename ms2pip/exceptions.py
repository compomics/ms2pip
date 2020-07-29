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


class PeptideTooLongError(Exception):
    """The passed peptide is too long to be processed."""
