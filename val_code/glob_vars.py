"""
globals.py

File that holds some important static dictionary look-up tables, and some useful
functions used repeatedly in the codebase.

"""

import numpy as np
import pandas as pd

from scipy.special import logit


"""
get_prec_df
    Reads in the precinct df given the state and the year.
    NOTE: also adds in normalized candidate cfscores

    @param: state, year, bool value of whether want CD or not

"""
def get_prec_df(state, year, add_cd=False, alt=False):
    if year == None:
        return None

    if state == 'ne':
        state_lst = ['tx', 'ny']
    else:
        state_lst = [state]

    prec_df_lst = []
    for state in state_lst:
        precdir = '../data/%s_data/data/%s_%s' % (state, state, year)

        if alt:
            precfile = '%s/precline_%s_alt_%s.csv' % (precdir, state, year)
        else:
            precfile = '%s/precline_%s_house_%s.csv' % (precdir, state, year)

        prec_df = pd.read_csv(precfile)

        # add in normalized candidate cfscores
        # [-2, 2] --> [0, 1] --> logit() --> [-inf, +inf]
        prec_df['cf_score_0'] = prec_df['cf_score_0'].astype(float)
        prec_df['cf_score_1'] = prec_df['cf_score_1'].astype(float)
        prec_df['cf_norm_0'] = logit(((prec_df['cf_score_0'] + 2) / 4.0))
        prec_df['cf_norm_1'] = logit(((prec_df['cf_score_1'] + 2) / 4.0))

        if add_cd:
            prec_cd_file = '%s/%s_%s_prec_cd.csv' % (precdir, state, year)
            prec_df = pd.merge(prec_df, pd.read_csv(prec_cd_file), how='left',
                    on='geoid')

        prec_df['state'] = state
        prec_df['geoid'] = prec_df['geoid'].astype('str')
        prec_df_lst.append(prec_df)

    return pd.concat(prec_df_lst)


next_year_dict = {
    '2006': '2008',
    '2008': '2010',
    '2010': None
}

alt_exists_dict = {
    'tx': {
        '2006': True,
        '2008': True,
        '2010': False,
    },
    'ny': {
        '2006': True,
        '2008': True,
        '2010': True,
    },
    'oh': {
        '2006': True,
        '2008': True,
        '2010': True,
    }
}

state_name_dict = {
    'tx': 'Texas',
    'ny': 'New York',
    'oh': 'Ohio'
}


# maps both kinds of state names to the lowercase way we like
state_cces_dict = {
    'Texas': 'tx',
    'New York': 'ny',
    'Ohio': 'oh',
    'TX': 'tx',
    'NY': 'ny',
    'OH': 'oh'
}


state_code_dict = {
    'tx': 48,
    'ny': 36,
    'oh': 39
}

# dictionaries that have hard-coded the string to integer values of CCES survey
cces_ideology_dict = {
    'very liberal': 1,
    'liberal': 2,
    'moderate': 3,
    'conservative': 4,
    'very conservative': 5,
    'not sure': -1,
    'skipped': -1,
    'not asked': -1
}

cces_self_p_dict = {
    'Very Liberal': 1,
    'Liberal': 2,
    'Somewhat Liberal': 3,
    'Middle of the Road': 4,
    'Somewhat Conservative': 5,
    'Conservative': 6,
    'Very Conservative': 7,
    'Not Sure': -1,
    'Skipped': -1,
    'Not Asked': -1,
    np.nan: -1
}

