"""
lin_reg_pred.py

Code for cross-validation leave-one-out procedure. Take precincts of one
district, train a regression model to predict vote shares given whatever
metric we are testing against (surveys, MRP, previous year's vote share, etc.),
and see how it compares to using our method on the remaining district.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

from glob_vars import *
from sklearn import linear_model, cross_validation

### important variables ###
PROB_TRAINED = 0.9
NUM_ITER = 100


"""
init
    Gets the dataframe of precinct-level election results for the user's
    specified election we're trying to predict
    Gets the dataframe of the feature info, all by district.
"""
def init(path_to_results='../'):
    # get input
    print 'Please input data about the election you are trying to predict.'
    state = raw_input('State: ')
    year = raw_input('Year: ')
    num_clust = raw_input('Number of clusters: ')
    prec_dist = raw_input('Prec dist: ')

    # pulling feature directory information
    print 'Please select folder of dist_comp.csv you want as input: '
    feature_dir = '%sresults/%s/%s_clusters/mh_%s/%s/comparison_charts' % \
            (path_to_results, state, num_clust, year, prec_dist)

    if not(os.path.exists('%s/dist_comp.csv' % feature_dir)):
        print '[error] dist_comp.csv is not in dir: %s' % feature_dir
        assert(False)

    alt_df = pd.DataFrame()
    if alt_exists_dict[state][year]:
        alt_df = get_prec_df(state, year, True, alt=True)

    next_year = next_year_dict[year]
    cong_df = pd.DataFrame()
    if next_year != None:
        cong_df = get_prec_df(state, next_year, True, alt=False)

    return alt_df, cong_df, feature_dir


"""
format_prediction_data
    Creates training and testing data set given the input/output election data
"""
def format_prediction_data(vote_by_prec_df, feature_dir, metric_colname):
    # data of election we are trying to predict
    vote_by_dist_df = vote_by_prec_df.groupby(['state', 'congdist'],
            as_index=False).sum()
    vote_by_dist_df['tot_votes'] = vote_by_dist_df['tot_votes'].astype(float)
    output_df = pd.DataFrame({
        'congdist': vote_by_dist_df['congdist'],
        'state': vote_by_dist_df['state'],
        'tot_votes': vote_by_dist_df['tot_votes'],
        'dem_vote_share': vote_by_dist_df['num_votes_0'] / \
                vote_by_dist_df['tot_votes'],
    })

    # data of past election we are using as input for prediction
    # note - already checked that dist_comp exists here
    input_df = pd.read_csv('%s/dist_comp.csv' % feature_dir)
    if not(metric_colname in input_df.columns):
        print '[error] metric column name is not in the dataframe'
        assert(False)
    input_df = input_df[['congdist', 'state', metric_colname]]

    # combining input and output data for the model
    combined_df = pd.merge(output_df, input_df, how='left',
            on=['congdist', 'state'])

    # split data 90/10
    combined_df = combined_df.dropna().reset_index()
    return combined_df


"""
train_model
    Trains a supervised learning model (for now only linear regression) to
    approximate given a specific metric the vote share of the test dataset.

    # metric eg: surveys, MRP, previous year's vote share, etc
"""
def train_model(training_df, metric_colname):
    lr = linear_model.LinearRegression()

    # [ metric ] -> vote share
    lr.fit(training_df[metric_colname][:,np.newaxis],
            training_df['dem_vote_share'])
    return lr


"""
eval_results
    Evaluates the error term and plots the comparison
"""
def eval_results(testing_df):
    per_diff_arr = testing_df['dem_vote_share'] - testing_df['pred_vote_share']
    return sum([x**2 for x in per_diff_arr])


"""
sanity_check
    Plots things just in case.
"""
def sanity_check(testing_df):
    plt.scatter(testing_df['dem_vote_share'], testing_df['pred_vote_share'])
    plt.show()


"""
run_cross_val
    Function to generalize the linear regression portion when there are a small
    number of data points to use as prediction (eg: 30) and the training dataset
    is very small

    > specifically, does a "LeaveOneOut" cross-validation idea
"""
def run_cross_val(combined_df, metric_colname, outdir, file_prefix):
    dist_arr = []
    error_arr = []
    num_data_rows = combined_df.shape[0]
    data_sets = cross_validation.LeaveOneOut(num_data_rows)

    for train_i, test_i in data_sets:
        training_df = combined_df.loc[train_i.tolist(),
                [metric_colname, 'dem_vote_share']]
        testing_df = combined_df.loc[test_i.tolist(),
                [metric_colname, 'dem_vote_share']]

        # train a regression model (make this generalizable)
        lr = train_model(training_df, metric_colname)

        # compute vote share of test data (remaining 10) and compare results
        testing_df['pred_vote_share'] = lr.predict(
                testing_df[metric_colname][:,np.newaxis])

        dist_arr.append(test_i[0])
        error_arr.append(eval_results(testing_df))

    out_df = pd.DataFrame({
        'error': error_arr,
        'congdist': dist_arr
    })
    out_df.to_csv('%s/%scross_val_pred_comp.csv' % (outdir, file_prefix))

    avg_error = sum(error_arr) / num_data_rows
    standard_error = np.std(error_arr) / np.sqrt(num_data_rows)
    return avg_error, standard_error


"""
run_lin_reg
    Function to genearlize the linear regression portion
"""
def run_lin_reg(combined_df, metric_colname):
    tot_error = 0

    for i in xrange(NUM_ITER):
        combined_df['training_set'] = np.random.choice([0, 1],
                size=len(combined_df.index),
                p=[1-PROB_TRAINED, PROB_TRAINED])
        training_df = combined_df[combined_df['training_set'] == 1]
        testing_df = combined_df[combined_df['training_set'] == 0]

        # train a regression model (make this generalizable)
        lr = train_model(training_df, metric_colname)

        # compute vote share of test data (remaining 10) and compare results
        testing_df['pred_vote_share'] = lr.predict(
                testing_df[metric_colname][:,np.newaxis])

        tot_error += eval_results(testing_df)

    avg_error = tot_error / NUM_ITER
    return avg_error


def supervised_model_pred(alt_df, cong_df, feature_dir, metric_colname,
    csv_writer=None, next_year=None):
    if len(cong_df.index) > 0:
        combined_df = format_prediction_data(cong_df,
                feature_dir, metric_colname)
        avg_error, standard_error = run_cross_val(combined_df, metric_colname,
                feature_dir, '')

    if len(alt_df.index) > 0:
        combined_df = format_prediction_data(alt_df,
                feature_dir, metric_colname)

        avg_error, standard_error = run_cross_val(combined_df, metric_colname,
                feature_dir, 'alt_')

    if csv_writer != None:
        csv_writer.writerow(["mrp_cross", avg_error, standard_error, next_year])



def main():
     # pull in an election cycle data based on user preferences
    alt_df, cong_df, feature_dir = init()

    # aggregate input and output data sources
    metric_colname = raw_input('Metric Colname (eg: mrp_estimate): ')
    supervised_model_pred(alt_df, cong_df, feature_dir, metric_colname)


if __name__ == "__main__":
    main()

