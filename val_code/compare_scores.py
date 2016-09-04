"""
compare_scores.py

Validation code related to comparing other results to our own.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from glob_vars import *
from lin_reg_pred import *
from pred import *
from scipy.stats import norm


"""
compare_cces_helper
    Function that helps construct the cces dataframes for compare_cces function.

"""
def compare_cces_helper(cces_df, col_dict, state_str, year_str):
    state_lst = []
    inclBoth = (state_str == 'ne')
    if year_str == '2006':
        if inclBoth:
            state_lst = ['tx'.upper(), 'ny'.upper()]
        else:
            state_lst = [state_str.upper()]
    else:
        if inclBoth:
            state_lst = [state_name_dict['tx'], state_name_dict['ny']]
        else:
            state_lst = [state_name_dict[state_str]]

    cces_df = cces_df[cces_df[col_dict['state']].isin(state_lst)]

    # converting all states to lowercase two-letter abbreviations
    cces_df.loc[:,'state'] = [state_cces_dict[x] for x in cces_df.loc[:,(col_dict['state'])]]
    clean_cces_df = pd.DataFrame({
        'congdist': cces_df[col_dict['congdist']],
        'ideo': cces_df[col_dict['ideo']],
        'self_p': cces_df[col_dict['self_p']],
        'state': cces_df['state'],
        'party': cces_df[col_dict['party']],
    })
    return clean_cces_df

"""
party_name_to_int
    Converts the name of the political party to an int so that when we take the
    district "mean" of the value, we will have the percentage of Democrats.

    Note: we also add 0.5 for any independent/other because we assume that they
    would eventually have to pick one of the two main candidates in the election.
"""
def party_name_to_int(party_str):
    if party_str == 'Democrat':
        return 1
    elif party_str == 'Republican':
        return 0
    else:
        return 0.5


"""
get_cces_df
    Gets the survey data of CCES in a cleaner, subsetted format.
"""
def get_cces_df(state_str, year_str, path_to_val):
    # read in df and subset by state
    cces_df = pd.read_csv('%s/survey_data/CCES_%s/cces_%s.csv' % (path_to_val,
        year_str, year_str), low_memory=False)
    if year_str == '2006':
        col_dict = {
            'state': 'v1002',
            'congdist': 'v1003',
            'ideo': 'v2021',
            'self_p': 'v3042',
            'party': 'v3005',
        }

        cces_df = compare_cces_helper(cces_df, col_dict, state_str, year_str)
        cces_df = cces_df[cces_df['self_p'] != 'Don\'t know']
    elif year_str == '2008':
        col_dict = {
            'state': 'V206',
            'congdist': 'V250',
            'ideo': 'V243',
            'self_p': 'CC317a',
            'party': 'CC307',
        }
        cces_df = compare_cces_helper(cces_df, col_dict, state_str, year_str)
    elif year_str == '2010':
        col_dict = {
            'state': 'V206',
            'congdist': 'V276',
            'ideo': 'V243',
            'self_p': 'CC334A',
            'party': 'V212a',
        }
        cces_df = compare_cces_helper(cces_df, col_dict, state_str, year_str)
        cces_df['self_p'] = [cces_self_p_dict[x] for x in cces_df['self_p']]
    else:
        print 'Cannot find matching CCES survey dataset. ):'
        assert(False)

    # taking average of survey responses on ideology
    cces_df = cces_df.dropna(0)
    cces_df['ideo'] = [cces_ideology_dict[x.lower()] for x in cces_df['ideo']]

    # taking out invalid values
    cces_df = cces_df[cces_df['ideo'] > 0]

    cces_df['self_p'] = cces_df['self_p'].astype(int)
    cces_df = cces_df[cces_df['self_p'] <= 100]

    # converting political party to numbers (later, mean = % of dem + 0.5 of
    # other)
    cces_df['party'] = [party_name_to_int(x) for x in cces_df['party']]

    return cces_df


"""
compare_cces
    Function to compare our results with CCES survey results of district-level
    preferences.
"""
def compare_cces(dist_df, state_str, year_str, path_to_val):
    cces_df = get_cces_df(state_str, year_str, path_to_val)

    cces_group_df = cces_df.groupby(['congdist', 'state']).mean()

    cces_group_df = pd.merge(dist_df, cces_group_df, how='left',
            left_on=['congdist', 'state'], right_index=True)
    cces_group_df.reset_index(inplace=True)
    return cces_group_df


"""
compare_warshaw
    Function to compare results with Warshaw's MRP results (cw_estimates.csv).

"""
def compare_warshaw(dist_df, year_str, path_to_val):
    dist_df['fips'] = dist_df[['state', 'congdist']].apply(lambda x:
            state_code_dict[x['state']] * 100 + x['congdist'], axis=1)

    warshaw_df = pd.read_csv('%s/cw_estimates.csv' % path_to_val)

    # compare the two pieces of data
    dist_df = pd.merge(dist_df, warshaw_df, how='left', on='fips')
    return dist_df


"""
compare_by_district
    Function that compares our estimates to related works, specifically on
    district-level political preference approximations.

    @param: state_str, year_str
    @ret: distmean_df to be written out to proper csv
"""
def compare_by_district(state_str, year_str, indir, path_to_val):
    distmean_df = pd.read_csv('%s/dist_means.csv' % indir)

    # cces survey results
    distmean_df = compare_cces(distmean_df, state_str, year_str, path_to_val)

    # comparing with warshaw (mrp_estimate - one estimate for the decade?)
    distmean_df = compare_warshaw(distmean_df, year_str, path_to_val)

    # return to write this out to csv file to be plot in ggplot
    return distmean_df


"""
compare_pred_tests
    Function that does a lot of tests on baseline and prediction stuff,
    generally relating our data to other election results.

"""
def compare_pred_tests(state, year, indir, alt, dist_df, feature_dir,
    csv_writer):
    prec_df = get_prec_df(state, year, add_cd=True)

    dist_asst_file = '%s/prec_asst_cd.csv' % indir
    asst_df = pd.read_csv(dist_asst_file)
    asst_df.loc[:,'geoid'] = asst_df['geoid'].astype('str')

    prec_df = pd.merge(prec_df, asst_df,
            how='left', on=['geoid', 'congdist'],
            suffixes=('', '_x'))

    param_df = pd.read_csv('%s/inferred_param_df.csv' % indir)

    alt_df = pd.DataFrame()
    cong_df = pd.DataFrame()
    if alt:
        next_year = '%s Alternative' % year
        next_prec_df = get_prec_df(state, year, add_cd=True, alt=alt)
        alt_df = next_prec_df

        merged_prec_df = pd.merge(prec_df, next_prec_df, how='left',
                on=['geoid', 'state'], suffixes=('_old', '_new'))
        merged_prec_df['congdist'] = merged_prec_df['congdist_new']
    else:
        # next year election info loaded in
        next_year = next_year_dict[year]
        next_prec_df = get_prec_df(state, next_year, add_cd=True, alt=alt)
        cong_df = next_prec_df

        merged_prec_df = pd.merge(prec_df, next_prec_df, how='left',
                on=['geoid', 'state'], suffixes=('_old', '_new'))

        # setting up for predicting the upcoming
        merged_prec_df['congdist'] = merged_prec_df['congdist_new']

    merged_prec_df.dropna(axis = 0, inplace = True)

    # print '++> Now considering %d out of %d-%d precincts for prediction.' % \
        (len(merged_prec_df.index), len(prec_df.index), len(next_prec_df.index))
    results_df = pred_one_year(merged_prec_df, param_df, year, next_year,
            dist_df, csv_writer)

    # also predicting MRP cross-val technique in here
    supervised_model_pred(alt_df, cong_df, feature_dir, 'mrp_estimate',
        csv_writer, next_year)

    return results_df


"""
init()
    Function that initializes the input and output directories.
"""
def init(path_to_results):
    # state
    state = raw_input('State: ')

    # year
    year = raw_input('Year: ')

    # prec_dist
    prec_dist = raw_input('Prec Dist: ')

    # num_cluster
    num_clust = raw_input('Number of Clusters: ')


    # directory containing info about inferred param (distmeans, asstvect, etc.)
    input_dir = '%sresults/%s/%s_clusters/mh_%s/%s/charts' % (path_to_results,
            state, num_clust, year, prec_dist)

    outdir = '%sresults/%s/%s_clusters/mh_%s/%s' % \
            (path_to_results, state, num_clust, year, prec_dist)

    return state, year, input_dir, outdir


"""
create_qq
    Function to create QQ plots comparing the distributions of preferences with
    survey data.

    @param: state_str, year_str, indir
    @ret: stores QQ plots
"""
def create_qq(state, year, indir, outdir):
    # get precinct + asst data
    prec_df = get_prec_df(state, year, add_cd=True)

    dist_asst_file = '%s/prec_asst_cd.csv' % indir
    asst_df = pd.read_csv(dist_asst_file)
    asst_df['geoid'] = asst_df['geoid'].astype('str')

    prec_df = pd.merge(prec_df, asst_df,
            how='left', on=['geoid', 'congdist'])
    tot_pop = sum(prec_df['tot_votes'])

    # get param data
    param_df = pd.read_csv('%s/inferred_param_df.csv' % indir)

    # get survey data
    survey_df = pd.read_csv('survey_data/CCES_%s/%s_cces_%s.csv' % (year,
        state, year))
    num_survey_resp = len(survey_df.index)

    # get # samples from prec propto. prec pop
    asst_arr = []
    for i in xrange(num_survey_resp):
        asst_arr.append(np.random.choice(prec_df['asst'], size=1, replace=True,
                p=prec_df['tot_votes'] / tot_pop)[0])

    pref = []

    i = 0
    # for each survey sample
    for asst in asst_arr:
        # get distribution of precinct (cluster)
        mu = param_df.loc[asst, 'mu']
        sig = param_df.loc[asst, 'sig']

        # randomly select pref from precinct dist
        pref.append(np.random.normal(mu, sig))

    out_df = pd.DataFrame({
        'asst': asst_arr,
        'pref': pref,
        'ideo': survey_df['ideo'],
        'self_p': survey_df['self_p']
    })

    # write survey and inferred dist to csv to create qqplot in R
    out_df.to_csv('%s/qqplot_data.csv' % outdir, index=False)


"""
compare_scores_main()
    Function to compare our inferred district means values with other works'
    district preference approximations (originally the main() function for the
    compare_scores.py file)

    @param: state, year
    @ret: dist_comp.csv - file with comparison to Warshaw/MRP, CCES survey
          pred_comp.csv - file with next election cycle, predicted results
          alt_pred_comp.csv - file with alternative election but same year,
                              and predicted results
          pred_error.csv - file with standard error of each prediction task
"""
def compare_scores_main(state, year, indir, outdir, path_to_val):
    outdir = '%s/comparison_charts' % outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print 'Comparing inferred scores with related works.'
    # compare by district
    dist_df = compare_by_district(state, year, indir, path_to_val)
    dist_fname = '%s/dist_comp.csv' % outdir
    dist_df.to_csv(dist_fname, index=False)

    pred_fname = ''
    alt_pred_fname = ''

    compare_fname = '{}/pred_error.csv'.format(outdir)
    compare_file = open(compare_fname, 'w+')
    compare_csv = csv.writer(compare_file, delimiter=',')
    compare_csv.writerow(["method", "error", "standard_error", "election_cycle"])

    # compare baselines and prediction
    if next_year_dict[year] != None:
        print 'Comparing with Upcoming Election'
        results_df = compare_pred_tests(state, year, indir, False, dist_df,
                outdir, compare_csv)
        pred_fname = '%s/pred_comp.csv' % outdir
        results_df.to_csv(pred_fname, index=False)

    if alt_exists_dict[state][year]:
        print 'Comparing with Alternative Election in Same Year'
        results_df = compare_pred_tests(state, year, indir, True, dist_df,
                outdir, compare_csv)
        alt_pred_fname = '%s/alt_pred_comp.csv' % outdir
        results_df.to_csv(alt_pred_fname, index=False)

    compare_file.close()
    return {
        'dist': dist_fname,
        'pred': pred_fname,
        'alt-pred': alt_pred_fname,
        'error': compare_fname
    }


def main():
    state_str, year_str, indir, outdir = init('../')
    compare_scores_main(state_str, year_str, indir, outdir, '.')

if __name__ == "__main__":
    main()

