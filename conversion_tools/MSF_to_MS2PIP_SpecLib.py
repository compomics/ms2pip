"""
## MSF and MGF to MS2PIP SpecLib
Reads an MSF file (SQLite DB), combines it with the matched (multiple) MGF files and writes a spectral library as 1 MS2PIP PEPREC and MGF file. Filters by a given FDR threshold, using q-values calculated from decoy hits or from Percolator.

*MSF_to_MS2PIP_SpecLib.py*
**Input:** MSF and MGF files
**Output:** Matched PEPREC and MGF file

```
usage: MSF_to_MS2PIP_SpecLib.py [-h] [-s MSF_FOLDER] [-g MGF_FOLDER]
                                [-o OUTNAME] [-f FDR_CUTOFF] [-p] [-c]

Convert Sequest MSF and MGF to MS2PIP spectral library.

optional arguments:
  -h, --help            show this help message and exit
  -s MSF_FOLDER, --msf MSF_FOLDER
                        Folder with Sequest MSF files (default: "msf")
  -g MGF_FOLDER, --mgf MGF_FOLDER
                        Folder with MGF spectrum files (default: "mgf")
  -o OUTNAME, --out OUTNAME
                        Name for output files (default: "SpecLib")
  -f FDR_CUTOFF, --fdr FDR_CUTOFF
                        FDR cut-off value to filter PSMs (default: 0.01)
  -p                    Use Percolator q-values instead of calculating them
                        from TDS (default: False)
  -c                    Combine multiple MSF files into one spectral library
                        (default: False)
```
"""

# --------------
# Import modules
# --------------
import argparse
import re
import sqlite3
import pandas as pd
from glob import glob
from os import getcwd


# ----------------------------------------------------------------
# Calculate q-Value from PeptideScores using Target-Decoy strategy
# ----------------------------------------------------------------
def CalcQVal(Peptides, conn):
    PeptideScores = pd.read_sql("select * from 'PeptideScores';", conn)
    PeptideScores_Decoy = pd.read_sql("select * from 'PeptideScores_Decoy';", conn)
    if len(PeptideScores_Decoy) == 0:
        print("No decoy PSMs found: cannot calculate q-value.")
        exit(1)
    PeptideScores['Decoy'] = False
    PeptideScores_Decoy['Decoy'] = True
    PeptideScores = PeptideScores.append(PeptideScores_Decoy)

    PeptideScores.sort_values('ScoreValue', ascending=False, inplace=True)
    PeptideScores['CumSum_Decoys'] = PeptideScores['Decoy'].cumsum()
    PeptideScores['CumSum_Targets'] = (~PeptideScores['Decoy']).cumsum()
    PeptideScores['q-Value'] = PeptideScores['CumSum_Decoys'] / PeptideScores['CumSum_Targets']
    PeptideScores = PeptideScores[~PeptideScores['Decoy']][['PeptideID', 'q-Value']]

    Peptides = Peptides.merge(PeptideScores, on='PeptideID')

    return(Peptides)


# ------------------------------------------------------
# Get Percolator q-value and PEP out of CustomDataFields
# ------------------------------------------------------
def GetPercolatorQVal(Peptides, conn):
    CustomDataFields = pd.read_sql("select FieldID, DisplayName from 'CustomDataFields';", conn)
    CustomDataFields.index = CustomDataFields['FieldID']
    CustomDataFields_dict = CustomDataFields['DisplayName'].to_dict()

    CustomDataPeptides = pd.read_sql("select * from CustomDataPeptides;", conn)
    CustomDataPeptides['FieldName'] = [CustomDataFields_dict[ID] for ID in CustomDataPeptides['FieldID']]
    CustomDataPeptides = CustomDataPeptides.pivot(values='FieldValue', index='PeptideID', columns='FieldName').reset_index()

    Peptides = Peptides.merge(CustomDataPeptides, on='PeptideID')

    return(Peptides)


# --------
# Get PTMs
# --------
def ConcatPTMs(series):
        return(series.str.cat(sep='|'))


def GetPTMs(Peptides, conn):
    AminoAcidModifications = pd.read_sql("select * from 'AminoAcidModifications'", conn)
    PeptidesAminoAcidModifications = pd.read_sql("select * from 'PeptidesAminoAcidModifications'", conn)
    PeptidesTerminalModifications = pd.read_sql("select * from 'PeptidesTerminalModifications'", conn)

    # Get amino acid IDs and names in dict for easy access
    AminoAcidModifications.index = AminoAcidModifications.AminoAcidModificationID
    AminoAcidModifications_dict = AminoAcidModifications['ModificationName'].to_dict()

    # Combine terminal and normal PTMs into one dataframe
    PeptidesTerminalModifications.columns = ['ProcessingNodeNumber', 'PeptideID', 'AminoAcidModificationID']
    PeptidesTerminalModifications['PositionMS2PIP'] = 0
    PeptidesAminoAcidModifications['PositionMS2PIP'] = PeptidesAminoAcidModifications['Position'] + 1
    Modifications = PeptidesAminoAcidModifications.append(PeptidesTerminalModifications, ignore_index=True)

    # Translate PTM IDs to names
    Modifications['ModificationName'] = [AminoAcidModifications_dict[ID] for ID in Modifications['AminoAcidModificationID']]

    # Concatenate PTMs to MS2PIP notation per peptide
    Modifications.sort_values('PositionMS2PIP', inplace=True)
    Modifications['NotationMS2PIP'] = Modifications['PositionMS2PIP'].map(str) + '|' + Modifications['ModificationName']
    MS2PIPmods = pd.DataFrame(Modifications.groupby('PeptideID')['NotationMS2PIP'].apply(ConcatPTMs))
    MS2PIPmods.rename(columns={'NotationMS2PIP': 'Modifications'}, inplace=True)
    MS2PIPmods.reset_index(inplace=True)
    Peptides = Peptides.merge(MS2PIPmods, on='PeptideID')

    return(Peptides)


# ------------------------------------------
# Get spectrum properties and RAW file names
# ------------------------------------------
def GetSpectrumProperties(Peptides, conn):
    cols = 'SpectrumID, Charge, Mass, LastScan, FirstScan, ScanNumbers, MassPeakID'
    SpectrumHeaders = pd.read_sql("select {} from SpectrumHeaders".format(cols), conn)

    FileInfos = pd.read_sql("select FileID, FileName from FileInfos", conn)
    FileInfos['FileName'] = FileInfos['FileName'].str.split('\\')
    FileInfos['FileName'] = [name[-1].split('.raw')[0].split('.mzML')[0] for name in FileInfos['FileName']]

    MassPeaks = pd.read_sql("select MassPeakID, FileID from MassPeaks;", conn)
    MassPeaks = MassPeaks.merge(FileInfos, on='FileID')

    SpectrumHeaders = SpectrumHeaders.merge(MassPeaks, on='MassPeakID')
    Peptides = Peptides.merge(SpectrumHeaders, on='SpectrumID')

    return(Peptides)


# -------------------------------------------------------
# Filter Peptide list for non-redundancy and low q-values
# -------------------------------------------------------
def FilterForSpecLib(Peptides, FDR_CutOff):
    # Filter on q-value
    len_before = len(Peptides)
    Peptides = Peptides[Peptides['q-Value'] <= FDR_CutOff].copy()
    len_after = len(Peptides)
    print("q-value cut-off {}: Kept {} out of {} peptides.".format(FDR_CutOff, len_after, len_before))

    # Remove duplicate peptides: keep the one with the highest q-value
    len_before = len(Peptides)
    Peptides.sort_values('q-Value', inplace=True)
    Peptides = Peptides[~Peptides.duplicated(['Sequence', 'Modifications', 'Charge'], keep='first')].copy()
    len_after = len(Peptides)
    print("Removed duplicate peptides: Kept {} out of {} peptides.".format(len_after, len_before))

    # Remove multiple peptides mapping to the same spectrum: keep the one with the highest q-value
    len_before = len(Peptides)
    Peptides = Peptides[~Peptides.duplicated(['SpectrumID', 'FileName'], keep='first')].copy()
    len_after = len(Peptides)
    print("Removed duplicate spectra: Kept {} out of {} peptides.".format(len_after, len_before))

    # Annotate tryptic peptides
    Peptides['Tryptic'] = ((Peptides['Sequence'].str.endswith('K')) | (Peptides['Sequence'].str.endswith('R')))

    # Set PeptideID as index
    Peptides.index = Peptides['PeptideID']

    return(Peptides)


# ------------------------------------------------------------------
# Scan multiple MGF files for spectra present in Peptides data frame
# ------------------------------------------------------------------
def ScanMGF(df_in, mgf_folder, outname='SpecLib.mgf'):
    with open(outname, 'w') as out:
        count_runs = 0
        count = 0
        runs = df_in['FileName'].unique()
        print("Scanning MGF files: {} runs to do. Now working on run: ".format(len(runs)), end='')
        for run in runs:
            count_runs += 1
            if count_runs % 10 == 0:
                print(str(count_runs), end='')
            else:
                print('.', end='')
            spec_dict = dict((v, k) for k, v in df_in[(df_in['FileName'] == run)]['FirstScan'].to_dict().items())

            # Parse file
            found = False
            with open('{}/{}.mgf'.format(mgf_folder, str(run)), 'r') as f:
                for i, line in enumerate(f):
                    if 'TITLE' in line:
                        ScanNum = int(re.split('scan=|"', line)[-2])
                        if ScanNum in spec_dict:
                            found = True
                            out.write("BEGIN IONS\n")
                            line = "TITLE=" + str(spec_dict[ScanNum]) + '\n'
                            count += 1
                    if 'END IONS' in line:
                        if found:
                            out.write(line + '\n')
                            found = False
                    if found and line[-4:] != '0.0\n':
                        out.write(line)

    print("\n{}/{} spectra found and written to new MGF file.".format(count, len(df_in)))


# ------------------------
# Write MS2PIP PEPREC file
# ------------------------
def WritePEPREC(Peptides, outname='SpecLib.peprec'):
    peprec = Peptides[['PeptideID', 'Modifications', 'Sequence', 'Charge']]
    peprec.columns = ['spec_id', 'modifications', 'peptide', 'charge']
    peprec.to_csv(outname, sep=' ', na_rep='-', index=None)


# ---------------
# Argument parser
# ---------------
def ArgParse():
    parser = argparse.ArgumentParser(description='Convert Sequest MSF and MGF to MS2PIP spectral library.')
    parser.add_argument('-s', '--msf', dest='msf_folder', action='store', default='msf',
                        help='Folder with Sequest MSF files (default: "msf")')
    parser.add_argument('-g', '--mgf', dest='mgf_folder', action='store', default='mgf',
                        help='Folder with MGF spectrum files (default: "mgf")')
    parser.add_argument('-o', '--out', dest='outname', action='store', default='SpecLib',
                        help='Name for output files (default: "SpecLib")')
    parser.add_argument('-f', '--fdr', dest='FDR_CutOff', action='store', default=0.01, type=float,
                        help='FDR cut-off value to filter PSMs (default: 0.01)')
    parser.add_argument('-p', dest='percolator', action='store_true', default=False,
                        help='Use Percolator q-values instead of calculating them from TDS (default: False)')
    parser.add_argument('-c', dest='combine_msf', action='store_true', default=False,
                        help='Combine multiple MSF files into one spectral library (default: False)')
    args = parser.parse_args()

    return(args)


# ----
# Run!
# ----
def run():
    args = ArgParse()
    msf_files = glob("{}/*.msf".format(args.msf_folder))
    msf_count = 0

    if args.combine_msf:
        Peptides = pd.DataFrame()
        for msf_filename in msf_files:
            msf_count += 1
            print("Adding MSF file {} of {}...".format(msf_count, len(msf_files)))
            conn = sqlite3.connect(msf_filename)
            Peptides_tmp = pd.read_sql("select * from Peptides;", conn)
            if args.percolator:
                Peptides_tmp = GetPercolatorQVal(Peptides_tmp, conn)
            else:
                Peptides_tmp = CalcQVal(Peptides_tmp, conn)
            Peptides_tmp = GetPTMs(Peptides_tmp, conn)
            Peptides_tmp = GetSpectrumProperties(Peptides_tmp, conn)
            Peptides = Peptides.append(Peptides_tmp)

        Peptides = FilterForSpecLib(Peptides, args.FDR_CutOff)

        outname_appendix = msf_files[0]
        outname_appendix = outname_appendix.replace('/', '')
        outname_appendix = outname_appendix.replace('.msf', '')

        ScanMGF(Peptides, args.mgf_folder, outname='{}_{}_Combined.mgf'.format(args.outname, outname_appendix))
        WritePEPREC(Peptides, outname='{}_{}_Combined.peprec'.format(args.outname, outname_appendix))

    else:
        for msf_filename in msf_files:
            msf_count += 1
            print("\nWorking on MSF file {} of {}...".format(msf_count, len(msf_files)))
            conn = sqlite3.connect(msf_filename)
            Peptides = pd.read_sql("select * from Peptides;", conn)
            if args.percolator:
                Peptides = GetPercolatorQVal(Peptides, conn)
            else:
                Peptides = CalcQVal(Peptides, conn)
            Peptides = GetPTMs(Peptides, conn)
            Peptides = GetSpectrumProperties(Peptides, conn)
            Peptides = FilterForSpecLib(Peptides, args.FDR_CutOff)

            outname_appendix = msf_filename
            outname_appendix = outname_appendix.replace('/', '')
            outname_appendix = outname_appendix.replace('.msf', '')

            ScanMGF(Peptides, args.mgf_folder, outname='{}_{}.mgf'.format(args.outname, outname_appendix))
            WritePEPREC(Peptides, outname='{}_{}.peprec'.format(args.outname, outname_appendix))

    print("Ready!")


if __name__ == "__main__":
    run()
