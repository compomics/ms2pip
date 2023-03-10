from peptides import AMINO_ACIDS


def get_feature_names():
    num_props = 4
    names = ["peplen", "charge"]
    for t in range(5):
        names.append("charge" + str(t))
    for t in range(num_props):
        names.append("qmin_%i" % t)
        names.append("q1_%i" % t)
        names.append("q2_%i" % t)
        names.append("q3_%i" % t)
        names.append("qmax_%i" % t)
    names.append("len_n")
    names.append("len_c")

    for a in AMINO_ACIDS:
        names.append("I_n_%s" % a)
        names.append("I_c_%s" % a)

    for t in range(num_props):
        for pos in ["p0", "pend", "pi-1", "pi", "pi+1", "pi+2"]:
            names.append("prop_%i_%s" % (t, pos))
        names.append("sum_%i_n" % t)
        names.append("q0_%i_n" % t)
        names.append("q1_%i_n" % t)
        names.append("q2_%i_n" % t)
        names.append("q3_%i_n" % t)
        names.append("q4_%i_n" % t)
        names.append("sum_%i_c" % t)
        names.append("q0_%i_c" % t)
        names.append("q1_%i_c" % t)
        names.append("q2_%i_c" % t)
        names.append("q3_%i_c" % t)
        names.append("q4_%i_c" % t)

    return names
