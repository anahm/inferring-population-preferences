"""
check_data.py

Checks for basic things to make sure quality of data still valid.

1.3.15
"""

import numpy.testing as npt
import pandas as pd

"""
check_same
    Function to check that there are no weird equal vote shares unless expected.

"""
def check_same(df):
    df = df[df['cf_score_1'] != 0]
    temp = df[df['num_votes_0'] == df['num_votes_1']]
    print 'Number of Identical Vote Share: ' + str(len(temp.index))


"""
check_left
    Function to check that the cf_score_0 < cf_score_1 is always maintained.
    This is _super_ important for the MH algorithm!

"""
def check_left(df):
    temp = df['cf_score_1'] - df['cf_score_0']
    assert((temp > 0).all())


"""
compare_orig
    Function to compare the current vote share in precline_* with the original
    dataset.
"""
def compare_orig(df, state, year, state_data_dir, rep_col, dem_col):
    orig_df = pd.read_csv('%s/data/orig_data/%s_precvote.csv' % \
            (state_data_dir, state))

    merge_df = pd.merge(df, orig_df, on='geoid', how='left')

    print merge_df[merge_df['num_votes_1'] != merge_df[rep_col]]

    npt.assert_array_equal(merge_df['num_votes_1'], merge_df[rep_col])
    npt.assert_array_equal(merge_df['num_votes_0'], merge_df[dem_col])

    print 'Vote shares properly ported over!'


"""
check_main()
    What prec_reformat can call.
"""
def check_main(infile, state, year, rep_col, dem_col):
    state_data_dir = '../data/%s_data' % state
    df = pd.read_csv(infile)

    check_same(df)
    check_left(df)
    compare_orig(df, state, year, state_data_dir, rep_col, dem_col)


def main():
    state = raw_input('State: ')
    year = raw_input('Year: ')

    data_dir = '../data/%s_data/data/%s_%s' % (state, state, year)

    infile = '%s/precline_%s_house_%s.csv' % (data_dir, state, year)
    rep_col = 't_USH_R_%s' % year
    dem_col = 't_USH_D_%s' % year

    check_main(infile, state, year, rep_col, dem_col)


if __name__ == "__main__":
    main()

