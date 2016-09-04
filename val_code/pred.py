"""
pred.py

Validation code related to using past inferred data to predict the vote share
for candidates in the future.

Note: there are now multiple possible prediction methods of election results
    # baseline := people vote for same party candidate as previous election
    # inferredData := people vote according to inferred preferences of voters
    # survey := people vote according to reported political party
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import norm, binom


"""
mixed_pred_cand_vote

    Predicts the number of votes of two candidates in the next election based on
    precinct distributions being mixed distributions based on cluster parameters
    and likelihood of matching a cluster.

    @param: prec_df, param_df
    @ret: pred_vote_dict -- (state, congdist) --> (num_votes_0, num_votes_1)
"""
def mixed_pred_cand_vote(prec_df, param_df):
    pred_vote_dict = {}

    num_clusters = len(param_df.index)
    print 'num_clusters: %d' % num_clusters

    # iterate per precinct
    num_prec = len(prec_df.index)
    for i, row in prec_df.iterrows():
        if row['cf_score_1_old'] == 0 or np.isnan(row['cf_score_1_new']):
            # Skipping uncontested candidates
            continue

        # get midpoint of 2 cand of next year's election
        mid_cf_old = (row['cf_norm_0_old'] + row['cf_norm_1_new']) / 2.0
        mid_cf_new = (row['cf_norm_0_new'] + row['cf_norm_1_new']) / 2.0

        # expected value -- most likely predicted vote share
        exp_val = 0.0

        # for every possible voter value
        tot_votes = int(np.round(row['tot_votes_new']))
        for i in xrange(tot_votes):
            # for every cluster
            sj_arr = binom.pmf(row['num_votes_0_old'], row['tot_votes_old'],
                    norm.cdf(mid_cf_old, param_df['loc'], param_df['scale'])) * \
                            param_df['theta']

            # vector of prob of votes given cluster asst
            prob_vector = sj_arr * binom.pmf(i, tot_votes,
                    norm.cdf(mid_cf_new, param_df['loc'], param_df['scale']))

            # normalizing constant for the S_j term
            norm_const = sum(sj_arr)

            # compute probability of # votes for c0 == i
            exp_val += (i * np.sum(prob_vector / norm_const))

        # fill in votes for the district..
        state = row['state']
        congdist = row['congdist']
        if not((state, congdist) in pred_vote_dict):
            pred_vote_dict[(state, congdist)] = [0] * 2

        # add values to pred_vote_dict
        pred_vote_dict[(state, congdist)][0] += exp_val
        pred_vote_dict[(state, congdist)][1] += (tot_votes - exp_val)

        if i % 100 == 0:
            print 'Precinct %d/%d' % (i, num_prec)

    return pred_vote_dict


"""
pred_cand_vote

    Predicts the number of votes the two candidates of the next election will
    receive based on the previous election results and inferred values.

    @param: prec_df that has the old and new elections merged together
            param_df
    @ret: pred_vote_dict -- (state, congdist) --> (num_votes_0, num_votes_1)
"""
def pred_cand_vote(prec_df, param_df):
    # now hashes by a tuple (state, congdist)
    pred_vote_dict = {}

    # iterate per precinct
    for i, row in prec_df.iterrows():
        if row['cf_score_1_old'] == 0 or np.isnan(row['cf_score_1_new']):
            # Skipping uncontested candidates
            continue

        # get midpoint of 2 cand of next year's election
        mid_cf = (row['cf_norm_0_new'] + row['cf_norm_1_new']) / 2.0

        # get precinct assignment (last row, col i)
        asst = int(row['asst'])
        loc = param_df.loc[asst, 'loc']
        scale = param_df.loc[asst, 'scale']

        # Assuming c0 < c1
        c0_vote_per = norm.cdf(mid_cf, loc=loc, scale=scale)
        c1_vote_per = 1 - c0_vote_per
        assert(c1_vote_per == (1 - norm.cdf(mid_cf, loc=loc, scale=scale)))

        # precinct next year's tot_votes
        tot_votes = row['tot_votes_new']

        # Rounding to whole numbers of votes
        c0_vote_num = np.round(c0_vote_per * tot_votes)
        c1_vote_num = np.round(c1_vote_per * tot_votes)

        # fill in votes for the district..
        state = row['state']
        congdist = row['congdist']
        if not((state, congdist) in pred_vote_dict):
            pred_vote_dict[(state, congdist)] = [0] * 2

        pred_vote_dict[(state, congdist)][0] += c0_vote_num
        pred_vote_dict[(state, congdist)][1] += c1_vote_num

    return pred_vote_dict


"""
compare_pred
    Function to compare the predicted vote share with the actual.

"""
def compare_pred(new_prec_df, cand_dict, next_year, predMethod,
        dist_df=None):
    cd_arr = []
    state_arr = []
    pred_arr = []
    act_arr = []
    per_diff_arr = []

    grouped_df = new_prec_df.groupby(['state', 'congdist'])
    for (state, cd), grp in grouped_df:
        if not((state, cd) in cand_dict):
            # Skipping uncontested candidates
            continue

        state_arr.append(state)
        cd_arr.append(cd)

        tot_votes = float(sum(grp['tot_votes_new']))

        if predMethod == "baseline":
            old_tot_votes = float(sum(grp['tot_votes_old']))
            pred_vote_0 = sum(grp['num_votes_0_old']) / old_tot_votes
        elif predMethod == "inferredData":
            pred_vote_0 = cand_dict[(state, cd)][0] / tot_votes
        elif predMethod == "survey":
            pred_vote_0 = dist_df.loc[dist_df.congdist == cd]['party'].values[0]
        elif predMethod == "mrp":
            # min val = -1.09, so need to shift up
            min_mrp_val = 1.09
            mrp_range = 1.58
            mrp_est = dist_df[dist_df.congdist == cd]['mrp_estimate'].values[0]
            pred_vote_0 = (mrp_est + min_mrp_val) / mrp_range
        else:
            print '[ERROR] Have not developed prediction given predMethod: %s' % \
                predMethod
        pred_arr.append(pred_vote_0)

        act_vote_0 = sum(grp['num_votes_0_new']) / tot_votes
        act_arr.append(act_vote_0)

        # compare actual vs. expected vote share
        diff = act_vote_0 - pred_vote_0
        per_diff_arr.append(diff)

    results_df = pd.DataFrame({
        'congdist': cd_arr,
        'state': state_arr,
        'approx_0': pred_arr,
        'approx_1': [1-x for x in pred_arr],
        'act_0': act_arr,
        'act_1': [1-x for x in act_arr],
        'per_diff': per_diff_arr,
        'winner': [0 if x > 0.5 else 1 for x in pred_arr]
    })

    # mean of squared error
    squared_err_arr = [x**2 for x in per_diff_arr]
    num_dist = len(per_diff_arr)
    mean_error = sum(squared_err_arr) / num_dist
    print str(mean_error)

    standard_error = np.std(squared_err_arr) / np.sqrt(num_dist)
    print standard_error

    results_df.sort_values(by='congdist', inplace=True)
    return results_df, mean_error, standard_error


"""
pred_one_year
    Function that predicts the vote shares of the next election based on the
    prior election inferred values.

    @param: merged_prec_df, param_dict, year/next_year as strings,
            dist_df of survey data info
"""
def pred_one_year(merged_prec_df, param_df, year, next_year, dist_df,
    csv_writer):
    # Note: Can also use: mixed_pred_cand_vote(merged_prec_df, param_df)
    pred_vote_dict = pred_cand_vote(merged_prec_df, param_df)

    pred_results_df, mean, std_err = compare_pred(merged_prec_df, pred_vote_dict,
            next_year, "inferredData")
    csv_writer.writerow(["inferred", mean, std_err, next_year])

    base_results_df, mean, std_err = compare_pred(merged_prec_df, pred_vote_dict,
            next_year, "baseline")
    csv_writer.writerow(["baseline", mean, std_err, next_year])

    merged_results_df = pd.merge(pred_results_df, base_results_df,
            how='left',
            on=['congdist', 'act_0', 'act_1', 'state'],
            suffixes=('', '_base'))

    survey_results_df, mean, std_err = compare_pred(merged_prec_df, pred_vote_dict,
            next_year, "survey", dist_df)
    csv_writer.writerow(["survey", mean, std_err, next_year])

    merged_results_df = pd.merge(merged_results_df, survey_results_df,
        how='left',
        on=['congdist', 'act_0', 'act_1', 'state'],
        suffixes=('', '_survey'))

    mrp_results_df, mean, std_err = compare_pred(merged_prec_df, pred_vote_dict,
            next_year, "mrp", dist_df)
    csv_writer.writerow(["mrp_basic", mean, std_err, next_year])

    merged_results_df = pd.merge(merged_results_df, mrp_results_df,
        how='left',
        on=['congdist', 'act_0', 'act_1', 'state'],
        suffixes=('_pred', '_mrp'))

    return merged_results_df

