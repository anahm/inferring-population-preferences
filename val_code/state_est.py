"""
state_est.py

Computes the prediction error term of a given method for all congressional
districts in all states in our data set. Also computes standard error of the
prediction error term.
"""

import csv
import pandas as pd
import numpy as np
import os

from glob_vars import *


"""
merge_files
    Function to create one pandas df given file names of csv data.
"""
def merge_files(dir_names, filename):
    if len(dir_names) == 0:
        return

    # read in data
    input_df_arr = []
    for indir in dir_names:
        input_df_arr.append(pd.read_csv('%s/%s' % (indir, filename)))

    # merge two files
    return pd.concat(input_df_arr)


"""
combine_error_terms
    Function to consolidate the prediction error terms of each congressional
    district in the state data given by "file_names." Returns nothing, but
    prints mean and standard error.
"""
def combine_error_terms(dir_names, alt_file_prefix, outcsv, pred_cycle):
    if len(dir_names) == 0:
        return
    pred_df = merge_files(dir_names, '%spred_comp.csv' % alt_file_prefix)

    pred_methods = ['_pred', '_base', '_survey', '_mrp']

    for suffix in pred_methods:
        # using column 'per_diff%s' % suffix
        pdiff_col = 'per_diff%s' % suffix
        squared_err_arr = [x**2 for x in pred_df[pdiff_col]]

        # take average
        error = np.mean(squared_err_arr)

        # take std / sqrt(num districts)
        standard_error = np.std(squared_err_arr) / np.sqrt(pred_df.shape[0])

        outcsv.writerow([suffix[1:], error, standard_error, pred_cycle,
          alt_file_prefix])


    # also need to compute the MSE of the cross-val technique
    cross_val_df = merge_files(dir_names,
        '%scross_val_pred_comp.csv' % alt_file_prefix)

    # take average
    error = np.mean(cross_val_df['error'])

    # take std / sqrt(num districts)
    standard_error = np.std(cross_val_df['error']) / np.sqrt(cross_val_df.shape[0])

    outcsv.writerow(["cross_val", error, standard_error, pred_cycle,
      alt_file_prefix])


def state_est_main(year, prec_dist, num_clust, path_to_results, outdir):
    states = ['tx', 'ny', 'oh']

    alt_files = []
    next_files = []
    for state in states:
        chart_dir = '%sresults/%s/%s_clusters/mh_%s/%s/comparison_charts' % \
                (path_to_results, state, num_clust, year, prec_dist)
        if not os.path.exists(chart_dir):
          continue
        print 'Adding in data from: {}'.format(chart_dir)

        # next election
        if next_year_dict[year] != None:
            next_files.append(chart_dir)

        # alternative election
        if alt_exists_dict[state][year]:
            alt_files.append(chart_dir)

    outfile_name = "{}/combined_state_pred.csv".format(outdir)
    outfile = open(outfile_name, 'w+')
    outcsv = csv.writer(outfile, delimiter=',')
    outcsv.writerow(["method", "error", "standard_error", "election_cycle",
        "is_alt_pred"])

    combine_error_terms(next_files, '', outcsv, next_year_dict[year])
    combine_error_terms(alt_files, 'alt_', outcsv, year)

    outfile.close()
    return outfile_name

def main():
    # year
    year = raw_input('Year: ')

    # prec_dist
    prec_dist = raw_input('Prec Dist: ')

    # num_cluster
    num_clust = raw_input('Number of Clusters: ')

    state_est_main(year, prec_dist, num_clust, '../', '.')


if __name__ == "__main__":
    main()

