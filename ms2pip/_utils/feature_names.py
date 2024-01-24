from ms2pip._utils.encoder import AMINO_ACIDS


def get_feature_names():
    """Generate a list of all prediction features in order."""
    aa_properties = ["basi", "heli", "hydro", "iso"]
    quartiles = ["min", "q1", "q2", "q3", "max"]
    ions = ["n", "c"]

    # Precursor properties
    names = ["p_length", "p_charge"]

    # One-hot encoded charge states
    for t in range(5):
        names.append(f"p_charge_{t + 1}")

    # Full precursor property quartiles
    for prop in aa_properties:
        for quartile in quartiles:
            names.append(f"p_{prop}_{quartile}")

    # Ion lengths
    for ion in ions:
        names.append(f"{ion}_length")

    # Ion amino acid counts
    for aa in AMINO_ACIDS:
        for ion in ions:
            names.append(f"{ion}_count_{aa}")

    for prop in aa_properties:
        # Properties for specific positions
        for pos in ["p0", "p-1", "pi-1", "pi", "pi+1", "pi+2"]:
            names.append(f"{pos}_{prop}")
        # Ion property quartiles
        for ion in ions:
            for metric in ["sum"] + quartiles:
                names.append(f"{ion}_{prop}_{metric}")

    return names
