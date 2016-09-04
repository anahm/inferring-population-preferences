"""
merge_orig.py

Merges all of the original New York datasets into a single one
(ny_precvote.csv). That way, we can format the data using the same scripts used
to format Texas election data.

Taken from PRECINCT-LEVEL ELECTION DATA on Election Data Archive Dataverse

"""

import pandas as pd

df_2006 = pd.read_csv("NY_2006.csv")[['vtdid', 'g2006_USH_dv', 'g2006_USH_rv', \
        'nopad_vtdid']]
df_2006['vtdid'] = df_2006['vtdid'].astype(int).astype('str')
df_2006['geoid'] = [int(x[2:]) for x in df_2006['vtdid']]

df_2008 = pd.read_csv("NY_2008.csv")[['county', 'vtd08', 'g2008_USH_dv', \
        'g2008_USH_rv', 'vtdid']]
df_2008['geoid'] = 10000 * df_2008['county'] + df_2008['vtd08']
df_2008.dropna(inplace=True)
df_2008['geoid'] = df_2008['geoid'].astype(int)
df_2008['nopad_vtdid'] = df_2008['vtdid']

df_2010 = pd.read_csv("NY_2010.csv")[['vtdid', 'g2010_USH_dv', 'g2010_USH_rv', \
        'nopad_vtdid']]
df_2010['vtdid'] = df_2010['vtdid'].astype(int).astype('str')
df_2010['geoid'] = [int(x[2:]) for x in df_2010['vtdid']]


df = pd.merge(df_2006, df_2010, how='outer', on='geoid')
df = pd.merge(df, df_2008, how='outer', on='geoid')


# making the column names consistent to tx_precvote.csv - easier for later
df.rename(columns={
    'g2006_USH_dv': 't_USH_D_2006',
    'g2006_USH_rv': 't_USH_R_2006',
    'g2008_USH_dv': 't_USH_D_2008',
    'g2008_USH_rv': 't_USH_R_2008',
    'g2010_USH_dv': 't_USH_D_2010',
    'g2010_USH_rv': 't_USH_R_2010',
}, inplace=True)

df['geoid'] = df['geoid'].astype(str)
df['nopad_vtdid'].fillna(0, inplace=True)
df['GEOID10'] = df['nopad_vtdid'].astype(int)

df.to_csv('ny_precvote.csv', index=False)
