"""
Calculate correlations from a pred_and_emp.csv file or calculate median
correlations by ion type from correlations.csv
"""


import argparse
import pandas as pd
from ms2pipC import calc_correlations


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("pae", metavar="<pae or correlations file>",
                        help="pred_and_emp.csv or correlations.csv file")
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    all_preds = pd.read_csv(args.pae)
    if args.pae.endswith('pred_and_emp.csv'):
        print('Computing correlations...')
        correlations = calc_correlations(all_preds)
        output_filename = args.pae.replace('pred_and_emp.csv', 'correlations.csv')
        print("Writing to {}".format(output_filename))
        correlations.to_csv("{}_correlations.csv".format(output_filename), index=True)
    else:
        correlations = all_preds
    print("Median correlations: ")
    print(correlations.groupby('ion')['pearsonr'].median())


if __name__ == '__main__':
    main()
