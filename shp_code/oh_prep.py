"""
oh_prep.py

Preparing the Ohio precinct-level voting results. Different file format from
NY/TX, so writing separate script.
"""

import numpy as np
import pandas as pd
import os.path

from prec_cd import dime_subset
from prec_reformat import convert_by_prec

def load_clean_data_2006(state, is_house_election, path_to_oh):
  oh_df = pd.read_csv("{}/orig_data/{}_precvote_{}.csv".format(
    path_to_oh, state, '2006'))
  print 'Loaded in {} rows for {} {} data.'.format(len(oh_df.index), state, 2006)

  # clean up the data
  column_lst = ['STATE_PRECINCT_CODE', 'US_CONGRESS_DISTRICT', 'COUNTY_NUM']
  if is_house_election:
    column_lst = column_lst + ['US_REP_D_NAME', 'US_REP_D_COUNT', 'US_REP_R_NAME',
      'US_REP_R_COUNT']

    # | geoid | num_votes_0 | num_votes_1 | tot_votes | cand_name_0 | cand_name_1 |
    prec_df = oh_df[column_lst]
    prec_df.rename(columns={
        'US_REP_D_NAME': 'cand_name_0',
        'US_REP_D_COUNT': 'num_votes_0',
        'US_REP_R_NAME': 'cand_name_1',
        'US_REP_R_COUNT': 'num_votes_1',
        'US_CONGRESS_DISTRICT': 'congdist'
      },
      inplace=True
    )
    prec_df = prec_df.dropna()
    prec_df.loc[:,'congdist'] = prec_df['congdist'].astype(int)
    # unique geoid = county-prec_alpha_id
    prec_df.loc[:,'geoid'] = prec_df['COUNTY_NUM'].astype(int).astype(str).str.zfill(3) + \
        '-' + prec_df['STATE_PRECINCT_CODE']

    prec_df['cand_name_0'].replace(regex=True, to_replace=r'\s\s',value=r' ',
        inplace=True)
    prec_df['cand_name_1'].replace(regex=True, to_replace=r'\s\s',value=r' ',
        inplace=True)

    prec_df.loc[:,('tot_votes')] = prec_df['num_votes_0'] + prec_df['num_votes_1']
    prec_outfile = "{}/data/oh_2006/oh_house_cand_2006.csv".format(path_to_oh)
    prec_df.to_csv(prec_outfile)
    print '{} {} house name data written to {}'.format(state, '2006', prec_outfile)

    # | (same) geoid | congdist |
    congdist_df = prec_df[['geoid', 'congdist']]
    congdist_outfile = "{}/data/oh_2006/oh_2006_prec_cd.csv".format(path_to_oh)
    congdist_df.to_csv(congdist_outfile)
    print '{} {} congdist data written to {}'.format(state, '2006',
        congdist_outfile)

    return prec_df
  else:
    # for 2006
    column_lst = column_lst + ['US_SENATE_D_BROWN', 'US_SENATE_R_DEWINE']

    sen_df = oh_df[column_lst]
    sen_df.rename(columns={
        'US_SENATE_D_BROWN': 'num_votes_0',
        'US_SENATE_R_DEWINE': 'num_votes_1'
      },
      inplace=True
    )

    # unique geoid = county-prec_alpha_id
    sen_df.loc[:,'geoid'] = sen_df['COUNTY_NUM'].astype(int).astype(str).str.zfill(3) + \
        '-' + sen_df['STATE_PRECINCT_CODE']

    sen_df.loc[:, ('cand_name_0')] = 'Sherrod Brown'
    sen_df.loc[:, ('cand_name_1')] = 'Mike DeWine'
    sen_df.loc[:, ('tot_votes')] = sen_df['num_votes_0'] + sen_df['num_votes_1']

    sen_outfile = "{}/data/oh_2006/oh_alt_cand_2006.csv".format(path_to_oh)
    sen_df.to_csv(sen_outfile)
    print '{} {} Senate Data written to {}'.format(state, '2006', sen_outfile)

    return sen_df


def load_clean_data_2008(state, is_house_election, path_to_oh):
  oh_df = pd.read_csv("{}/orig_data/{}_precvote_{}.csv".format(
    path_to_oh, state, '2008'))
  print 'Loaded in {} rows for {} {} data.'.format(len(oh_df.index), state, 2008)

  # COUNTY NUMBER,COUNTY NAME,STATE PRC CODE,PRECINCT NAME,"cand name (party)",
  # unique geoid = county-prec_alpha_id
  oh_df.loc[:,'geoid'] = oh_df['COUNTY NUMBER'].astype(int).astype(str).str.zfill(3) + \
      '-' + oh_df['STATE PRC CODE'].str[-3:]

  if is_house_election:
    house_df = (oh_df.iloc[:,8:]).set_index('geoid')

    cand_df = house_df.stack().reset_index()
    cand_df.rename(columns={'level_1': 'cand_name_party', 0:'num_votes'},
        inplace=True)

    # only want the relevant candidates to the county
    cand_df = cand_df[cand_df['num_votes'] > 0]

    cand_df["last_name"], cand_df["first_name"], _ = zip(
        *cand_df["cand_name_party"].str.split().tolist())
    cand_df.loc[:, "first_last_name"] = cand_df["first_name"] + " " + \
        cand_df["last_name"].str[:-1]

    # merge with congdist info
    congdist_file = "{}/orig_data/oh_2008_cand_key.csv".format(path_to_oh)
    congdist_df = pd.read_csv(congdist_file)[['cand_name_party', 'congdist']]
    cand_df = cand_df.merge(congdist_df, how='left', on='cand_name_party')

    prec_outfile = "{}/data/oh_2008/oh_house_cand_2008.csv".format(path_to_oh)

    # <county info>, cand_name_party, first_last_name (join with dime), num_votes
    cand_df.to_csv(prec_outfile)
    print '{} {} house name data written to {}'.format(state, '2008', prec_outfile)

    return cand_df
  else:
    # just hard-coding because faster
    pres_df = pd.DataFrame({
      'geoid': oh_df['geoid'],
      'num_votes_0': oh_df.iloc[:,6],
      'num_votes_1': oh_df.iloc[:,7],
      'tot_votes': oh_df.iloc[:,6] + oh_df.iloc[:,7],
      'midpoint': -0.335343058,
      'cf_score_0': -1.350869376, # obama
      'cf_score_1': 0.68018326 # mccain
    })

    # Subsetting columns to final DataFrame:
    prec_outfile = '{}/data/oh_2008/precline_oh_alt_2008.csv'.format(path_to_oh)
    pres_df.to_csv(prec_outfile)
    print 'pres data written to {}'.format(prec_outfile)
    return None


def load_clean_data_2010(state, is_house_election, path_to_oh):
  oh_df = pd.read_csv("{}/orig_data/{}_precvote_{}.csv".format(
    path_to_oh, state, '2010'))
  print 'Loaded in {} rows for {} {} data.'.format(len(oh_df.index), state, 2010)

  # COUNTY NUMBER,COUNTY NAME,STATE PRC CODE,PRECINCT NAME,"cand name (party)",
  oh_df['name_id'] = oh_df['PRECINCT_NAME']
  house_df = (oh_df.iloc[:,8:]).set_index('name_id')

  cand_df = house_df.stack().reset_index()
  cand_df.rename(columns={'level_1': 'cand_name_party', 0:'num_votes'},
      inplace=True)

  # only want the relevant candidates to the county
  cand_df = cand_df[cand_df['num_votes'] > 0]

  _, cand_df["congdist"], cand_df["last_first_name"] = zip(
      *cand_df["cand_name_party"].str.split(' - ').tolist())
  cand_df['congdist'] = cand_df['congdist'].str[-2:].astype(int)

  prec_df = oh_df.merge(cand_df, how='left', on='name_id')
  prec_df = prec_df.dropna()
  prec_df.loc[:,'congdist'] = prec_df['congdist'].astype(int)

  # unique geoid = county-prec_alpha_id
  prec_df.loc[:,'geoid'] = prec_df['COUNTY_NUM'].astype(int).astype(str).str.zfill(3) + \
      '-' + prec_df['PRECINCT_CODE']

  if is_house_election:
    prec_df['last'], prec_df['first'] = zip(
        *prec_df["last_first_name"].str.split(', ').tolist())
    prec_df.loc[:, "first_last_name"] = prec_df['first'] + ' ' + prec_df['last']

    prec_outfile = "{}/data/oh_2010/oh_house_cand_2010.csv".format(path_to_oh)

    # <county info>, cand_name_party, first_last_name (join with dime), num_votes
    prec_df.to_csv(prec_outfile)
    print '{} {} house name data written to {}'.format(state, '2010', prec_outfile)
    return prec_df
  else:
    prec_df = prec_df.groupby('geoid').first().reset_index()
    print prec_df.head()

    # just hard-coding because faster
    sen_df = pd.DataFrame({
      'geoid': prec_df['geoid'],
      'num_votes_0': prec_df["U.S. Senate - Fisher, Lee"],
      'num_votes_1': prec_df["U.S. Senate - Portman, Rob"],
      'tot_votes': prec_df["U.S. Senate - Fisher, Lee"] + \
          prec_df["U.S. Senate - Portman, Rob"],
      'midpoint': 0.1416439661365,
      'cf_score_0': -0.645953270361, # fisher
      'cf_score_1': 0.929241202634 # portman
    })

    print len(sen_df.index)

    # Subsetting columns to final DataFrame:
    prec_outfile = '{}/data/oh_2010/precline_oh_alt_2010.csv'.format(path_to_oh)
    sen_df.to_csv(prec_outfile)
    print 'sen data written to {}'.format(prec_outfile)

    return None


def load_clean_data(state, year, is_house_election, path_to_oh):
  if year == '2006':
    return load_clean_data_2006(state, is_house_election, path_to_oh)
  elif year == '2008':
    return load_clean_data_2008(state, is_house_election, path_to_oh)
  elif year == '2010':
    return load_clean_data_2010(state, is_house_election, path_to_oh)
  else:
    return None


def join_dime_2006(dime_df, oh_df):
  print dime_df.head()
  oh_df['cand_name_0'] = oh_df['cand_name_0'].str.upper()
  oh_df['cand_name_1'] = oh_df['cand_name_1'].str.upper()

  oh_df = oh_df.merge(dime_df, how='left', left_on='cand_name_0',
      right_on='first_last_name')
  oh_df.rename(columns={'cfscore': 'cf_score_0'}, inplace=True)

  oh_df = oh_df.merge(dime_df, how='left', left_on='cand_name_1',
      right_on='first_last_name')
  oh_df.rename(columns={'cfscore': 'cf_score_1'}, inplace=True)

  return oh_df


def join_dime(path_to_oh, year, e_type, oh_df):
  path_to_outdir = '{}/data/oh_{}'.format(path_to_oh, year)
  dime_file = '{}/{}_dime_subset_by_hand.csv'.format(path_to_outdir, e_type)

  df = None
  if year == '2006':
    dime_df = pd.read_csv(dime_file)[['first_last_name', 'cfscore']]
    df = join_dime_2006(dime_df, oh_df)

    # NOTE: have to drop uncontested candidates or dime data missing
    # 2006: 11789 --> 7462 because 5 candidates missing dime data
    # 2008: 4 candidates missing (22410 --> 19771)
    # 2010: 3 candidates missing (20217 --> 18548)
    df = df.dropna()
  elif year == '2008' or year == '2010':
    dime_df = pd.read_csv(dime_file)[['first_last_name', 'cf_score']]
    oh_df['first_last_name'] = oh_df['first_last_name'].str.upper()

    df = oh_df.merge(dime_df, how='left', on='first_last_name', indicator=True)
    df = df[df['_merge'] == 'both']

    outfile = convert_by_prec(df, 'oh', year, path_to_outdir)
    print 'Final prec output written to: {}'.format(outfile)

    # extra step to write out the congdist <--> prec file
    # | (same) geoid | congdist |
    congdist_outfile = "{}/data/oh_{}/oh_{}_prec_cd.csv".format(path_to_oh,
        year, year)
    congdist_df = pd.DataFrame({
        'geoid': df['geoid'],
        'congdist': df['congdist']
      })
    congdist_df.to_csv(congdist_outfile)
    print 'OH {} congdist data written to {}'.format(year, congdist_outfile)

    # don't want to write out file again
    df = None

  return df


def ohio_load_main(state, year, is_house_election, path_to_data='..'):
  # useful variables
  path_to_oh = '{}/data/oh_data'.format(path_to_data)
  e_type = 'house' if is_house_election else 'alt'

  # provide option to just output dime data
  if (raw_input('1 if output dime data, o/w election data: ') == '1'):
      output_dime_subset(state, year, is_house_election, path_to_oh)
      return


  # load and clean 'oh_precvote_<year>.csv' --> 'oh_<type>_cand_<year>.csv'
  cleaned_file = '{}/oh_{}/oh_{}_cand_{}.csv'.format(path_to_oh, year, e_type,
      year)
  oh_df = None
  if os.path.isfile(cleaned_file):
    oh_df = pd.read_csv(cleaned_file)
  else:
    oh_df = load_clean_data(state, year, is_house_election, path_to_oh)

  if type(oh_df) != pd.DataFrame:
    # means it was a senate one and already written out
    return

  # join with dime
  # Some of them are able to call my old function, so already written
  if type(oh_df) == pd.DataFrame:
    oh_df = join_dime(path_to_oh, year, e_type, oh_df)
    if len(oh_df) == 0:
      print '[ERROR]'
      print 'You probably need to go clean up the dime data candidate names..'
      assert(False)

    oh_df['midpoint'] = (oh_df['cf_score_0'] + oh_df['cf_score_1']) / 2.0

    # Subsetting columns to final DataFrame:
    prec_outfile = '{}/data/oh_{}/precline_oh_{}_{}.csv'.format(path_to_oh, year,
        e_type, year)

    oh_df[['cf_score_0', 'cf_score_1', 'geoid', 'num_votes_0', 'num_votes_1',
    'tot_votes', 'midpoint']].to_csv(prec_outfile)

    print 'Final prec data written to: {}'.format(prec_outfile)

  # done! (:
  return prec_outfile


def output_dime_subset(state, year, is_house_election, path_to_oh,
    path_to_data='..'):
  # join with dime data
  overall_dime_df = pd.read_csv('{}/data/dime_data.csv'.format(path_to_data))

  house_dime_df = dime_subset(overall_dime_df, state.upper(), year, 'house')[['Name',
    'cfscore', 'District', 'Party']]
  house_dime_df['District'] = house_dime_df['District'].str[2:].astype(int)
  sen_dime_df = dime_subset(overall_dime_df, state.upper(), year, 'senate')[['Name',
    'cfscore', 'District', 'Party']]
  sen_dime_df['District'] = -1
  dime_df = pd.concat([house_dime_df, sen_dime_df])


  dime_df["last_name"], dime_df["first_name"] = zip(*dime_df["Name"].str.split(', ').tolist())
  dime_df.loc[:, "first_last_name"] = dime_df["first_name"] + " " + dime_df["last_name"]

  e_type = 'house' if is_house_election else 'alt'
  dime_outfile = '{}/data/oh_{}/{}_dime_subset.csv'.format(path_to_oh, year, e_type)
  # dime_df.to_csv(dime_outfile)
  print 'Dime subset written to: {}'.format(dime_outfile)

  out_df = pd.DataFrame({
    'candname': dime_df['Name'],
    'cf_score': dime_df['cfscore'],
    'congdist': dime_df['District'],
    'party': dime_df['Party'],
    'first_last_name': dime_df['first_last_name']
  })
  out_outfile = '{}/data/oh_{}/{}_cand_{}.csv'.format(path_to_oh, year,
      state, year)
  out_df.to_csv(out_outfile)
  print 'cand_df written to: {}'.format(out_outfile)


def main():
  state = raw_input('State: ')
  assert(state == 'oh')
  year = raw_input('Year: ')

  is_house_election = raw_input('1 if House election, o/w senate: ') == '1'

  return ohio_load_main(state, year, is_house_election)


if __name__ == "__main__":
    main()

