"""
Scan MGF files in a given folder for spectra present in a given PEPREC file and
write those spectra to a new MGF file.
"""


__author__ = "Ralf Gabriels"
__credits__ = ["Ralf Gabriels", "Sven Degroeve", "Lennart Martens"]
__license__ = "Apache License, Version 2.0"
__version__ = "0.1"
__email__ = "Ralf.Gabriels@ugent.be"


import argparse
import mmap

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    use_tqdm_glob = False
else:
    use_tqdm_glob = True


def argument_parser():
    parser = argparse.ArgumentParser(
        description='Scan MGF files in a given folder for spectra present in a\
        given PEPREC file and write those spectra to a new MGF file.'
    )
    parser.add_argument(
        'mgf_folder', metavar="<mgf folder>",
        help='Folder containing MGF files to scan.'
    )
    parser.add_argument(
        "peprec_file", metavar="<peprec file>",
        help="MS2PIP PEPREC file that contains the peptide spectra to scan for.\
        For scan_mgf to work, an additional column `mgf_filename` that contains\
        the respective MGF filename is required."
    )
    # parser.add_argument(
    #     "--tqdm", action="store_true", dest='use_tqdm', default=False,
    #     help="Use tqdm to display progress bar."
    # )
    args = parser.parse_args()
    return args


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def scan_mgf(df_in, mgf_folder, outname='scan_mgf_result.mgf',
             filename_col='mgf_filename', spec_title_col='spec_id',
             use_tqdm=False):
    if df_in[filename_col].iloc[0][-4:] in ['.mgf', '.MGF']:
        file_suffix = ''
    else:
        file_suffix = '.mgf'

    with open(outname, 'w') as out:
        count_runs = 0
        count = 0
        runs = df_in[filename_col].unique()
        print("Scanning MGF files: {} runs to do. Now working on run: ".format(len(runs)), end='')
        for run in runs:
            count_runs += 1
            if count_runs % 10 == 0:
                print(str(count_runs), end='')
            else:
                print('.', end='')

            spec_dict = dict((v, k) for k, v in df_in[(df_in[filename_col] == run)][spec_title_col].to_dict().items())

            found = False
            current_mgf_file = '{}/{}{}'.format(mgf_folder, str(run), file_suffix)
            with open(current_mgf_file, 'r') as f:
                if use_tqdm:
                    mgf_iterator = tqdm(f, total=get_num_lines(current_mgf_file))
                else:
                    mgf_iterator = f
                for line in mgf_iterator:
                    if 'TITLE' in line:
                        title = line[6:].strip()
                        if title in spec_dict:
                            found = True
                            out.write("BEGIN IONS\n")
                            line = "TITLE=" + str(spec_dict[title]) + '\n'
                            count += 1
                    if 'END IONS' in line:
                        if found:
                            out.write(line + '\n')
                            found = False
                    if found and line[-4:] != '0.0\n':
                        out.write(line)

    print("\n{}/{} spectra found and written to new MGF file.".format(count, len(df_in)))


def main():
    args = argument_parser()
    peprec = pd.read_csv(args.peprec_file, sep=' ')
    scan_mgf(peprec, args.mgf_folder, use_tqdm=use_tqdm_glob)


if __name__ == '__main__':
    main()
