#!/usr/bin/env python3
"""
Convert MSP and SPTXT spectral library files

Writes three files: mgf with the spectra; PEPREC with the peptide sequences;
meta with additional metainformation.

Arguments:
    arg1 path to spectral library file
    arg2 prefix for spec_id
"""

import re
import sys


def main():
    AminoMass = {
        'A': 71.037114, 'C': 103.009185, 'D': 115.026943, 'E': 129.042593,
        'F': 147.068414, 'G': 57.021464, 'H': 137.058912, 'I': 113.084064,
        'K': 128.094963, 'L': 113.084064, 'M': 131.040485, 'N': 114.042927,
        'P': 97.052764, 'Q': 128.058578, 'R': 156.101111, 'S': 87.032028,
        'T': 101.047679, 'V': 99.068414, 'W': 186.079313, 'Y': 163.063329,
    }

    # Get arguments
    speclib_filename = sys.argv[1]
    title_prefix = sys.argv[2]

    # Open files
    filename = '.'.join(speclib_filename.split('.')[:-1])
    fpip = open(filename + '.peprec', 'w')
    fpip.write("spec_id modifications peptide charge\n")
    fmgf = open(filename + '.mgf', 'w')
    fmeta = open(filename + '.meta', 'w')

    PTMs = {}

    speclib_ext = speclib_filename.split('.')[-1]
    if speclib_ext in {'sptxt', 'SPTXT'}:
        speclib_format = 'sptxt'
    elif speclib_ext in {'msp', 'MSP'}:
        speclib_format = 'msp'
    else:
        print('Unknown spectral library format: {}'.format(speclib_ext))
        exit(1)

    print("Converting {} to MGF, PEPREC and meta file...".format(speclib_filename))

    specid = 1
    with open(speclib_filename) as f:
        peptide = None
        charge = None
        parentmz = None
        mods = None
        purity = None
        HCDenergy = None
        read_spec = False
        mgf = ""
        prev = 'A'
        # sys.stderr.write(prev)
        for row in f:
            if read_spec:
                line = row.rstrip().split('\t')

                # Read all peaks, so save to output files and set read_spec to False
                if not row[0].isdigit():
                    if peptide[0] != prev:
                        prev = peptide[0]
                        # sys.stderr.write(prev)

                    if mods[0] != '0':
                        tmp = mods[1:-1].replace("(","").replace(")",",").split(',')
                        m = ""
                        p = 0
                        for i in range(0, int(mods[0])):
                            if (tmp[p] == '0') & (tmp[p+2] == 'iTRAQ'):
                                m += '0|' + tmp[p+2] + '|'
                            else:
                                m += str(int(tmp[p])+1)+'|'+tmp[p+2] +'|'
                                # m += str(int(tmp2[0]) + 1) + '|' + tmp2[2] + peptide[int(tmp2[0])] + '|'
                            if not tmp[p+2] in PTMs:
                                PTMs[tmp[p+2]] = 0
                            PTMs[tmp[p+2]] += 1
                            p++3
                        fpip.write('{}{} {} {} {}\n'.format(title_prefix, specid, m[:-1], peptide, charge))

                    else:
                        fpip.write('{}{} - {} {}\n'.format(title_prefix, specid, peptide, charge))

                    fmeta.write('{}{} {} {} {} {} {}\n'.format(title_prefix, specid, charge, peptide, parentmz, purity, HCDenergy))

                    # THIS IS NOT A PROBLEM: MW is nothing
                    if mods[0] == '0':
                        if 'X' in peptide:
                            continue
                        tmp1 = 18.010601 + sum([AminoMass[x] for x in peptide])
                        tmp2 = (float(parentmz) * (float(charge))) - ((float(charge)) * 1.007825035)  # or 0.0073??
                        if abs(tmp1 - tmp2) > 0.5:
                            print(row)
                            print(".")

                    buf = "BEGIN IONS\n"
                    buf += "TITLE=" + title_prefix + str(specid) + '\n'
                    buf += "CHARGE=" + str(charge) + '\n'
                    buf += "PEPMASS=" + parentmz + '\n'
                    fmgf.write("{}{}END IONS\n\n".format(buf, mgf))

                    specid += 1
                    read_spec = False
                    mgf = ""

                else:
                    # Continue reading spectrum
                    # tt = float(line[1])
                    mgf += ' '.join([line[0], line[1]]) + '\n'
                    continue

            if row.startswith("Name:"):
                line = row.rstrip().split(' ')
                tmp = line[1].split('/')
                peptide = tmp[0].replace('(O)', '')
                if speclib_format == 'sptxt':
                    peptide = re.sub(r'\[\d*\]|[a-z]', '', peptide)
                charge = tmp[1].split('_')[0]
                continue

            if row.startswith("Comment:"):
                line = row.rstrip().split(' ')
                for i in range(1, len(line)):
                    if line[i].startswith("Mods="):
                        tmp = line[i].split('=')
                        mods = tmp[1]
                    if line[i].startswith("Parent="):
                        tmp = line[i].split('=')
                        parentmz = tmp[1]
                    if line[i].startswith("Purity="):
                        tmp = line[i].split('=')
                        purity = tmp[1]
                    if line[i].startswith("HCD="):
                        tmp = line[i].split('=')
                        HCDenergy = tmp[1].replace('eV', '')
                continue

            if row.startswith("Num peaks:") or row.startswith("NumPeaks:"):
                read_spec = True
                continue

    fmgf.close()
    fpip.close()
    fmeta.close()

    print("Finished!\nSpectral library contains {} peptides and these PTMs: {}".format(specid, PTMs))


if __name__ == "__main__":
    main()
