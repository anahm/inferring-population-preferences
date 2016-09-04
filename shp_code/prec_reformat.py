"""
prec_reformat.py

Taking state data and having each line be a precinct's voting results and candidate
cf-scores (rather than each line be each candidate per precinct.

| prec_id | cf_score_0 | num_votes_0 | cf_score_1 | num_votes_1 |

"""
import math
import numpy as np
import pandas as pd

from prec_cd import prec_cd_main
from check_data import check_main

def convert_by_prec(old_df, state, year, dirname):
    precs = []
    years = []
    cf_score_0 = []
    num_votes_0 = []
    cf_score_1 = []
    num_votes_1 = []

    # group by precinct (year assumed)
    for key, group in old_df.groupby(['geoid']):

        cf_iter = iter(group['cf_score'])
        votes_iter = iter(group['num_votes'])

        nxt_score = cf_iter.next()
        if math.isnan(nxt_score):
            nxt_score = 0
        cf_0 = nxt_score
        nv_0 = votes_iter.next()

        try:
            nxt_score = cf_iter.next()
            if math.isnan(nxt_score):
                nxt_score = 0
            cf_1 = nxt_score
            nv_1 = votes_iter.next()

            # enforcing the idea that cfscore0 < cfscore1
            precs.append(key)
            if cf_1 < cf_0:
                cf_score_0.append(cf_1)
                num_votes_0.append(nv_1)
                cf_score_1.append(cf_0)
                num_votes_1.append(nv_0)
            else:
                cf_score_0.append(cf_0)
                num_votes_0.append(nv_0)
                cf_score_1.append(cf_1)
                num_votes_1.append(nv_1)
        except StopIteration:
            # get rid of
            pass


    # use arrays to create dataframe
    new_df = pd.DataFrame({
        'cf_score_0': cf_score_0,
        'num_votes_0': num_votes_0,
        'cf_score_1': cf_score_1,
        'num_votes_1': num_votes_1,
        'geoid': precs},
        index=None)

    new_df['tot_votes'] = new_df['num_votes_0'] + new_df['num_votes_1']
    new_df['midpoint'] = (new_df['cf_score_0'] + new_df['cf_score_1']) / 2.0

    # write new dataframe out to csv
    outfile = '%s/precline_%s_house_%s.csv' % (dirname, state, year)
    new_df.to_csv(outfile)
    return outfile


"""
data_clean()
    Function to parse out certain types of data that are not useful in our
    results.

    # NOTE: overwrites the old file, since it is unnecessary

"""
def data_clean(precline_file):
    df = pd.read_csv(precline_file, index_col = 0)

    # remove all precincts with tot_votes == 0
    df = df[df['tot_votes'] > 0]

    # remove all uncontested candidates (cf_score_1 == 0)
    df = df[df['cf_score_1'] != 0]

    df.to_csv(precline_file, index=False)


"""
prec_reformat_main()
    Function that does the bulk of the original main function and can be called
    by the commandline.

    @param: state, year
    @return: location of new precline file
"""
def prec_reformat_main(state, year):
    prec_cd_main(state, year)

    csv_dir = '../data/%s_data/%s_%s' % (state, state, year)
    infile = '%s/%s_house_%s_final.csv' % (csv_dir, state, year)
    outfile = '%s/precline_%s_house_%s.csv' % (csv_dir, state, year)

    # read in file
    old_df = pd.read_csv(infile)

    convert_by_prec(old_df, state, year, csv_dir)
    data_clean(outfile)
    print 'Precinct data written to: %s' % outfile

    rep_col = 't_USH_R_%s' % year
    dem_col = 't_USH_D_%s' % year
    check_main(outfile, state, year, rep_col, dem_col)


def main():
    state = raw_input('State: ')
    year = raw_input('Year: ')
    prec_reformat_main(state, year)


if __name__ == "__main__":
    main()

