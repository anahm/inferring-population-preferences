"""
prec_cd.py

Using shapefiles of the state to link the precincts to the proper congressional
districts.

"""

# from osgeo import ogr, osr
# from pattern import web

import csv
import numpy as np
import pandas as pd
import requests
import os.path

"""
dime_subset

    Prints a subset of the dime data for just the specific year and elected
    position type.
"""
def dime_subset(dime_df, state, e_year, e_type):
    # get each candidate's cf-score from DIME data
    temptbl = dime_df[dime_df.cycle == int(e_year)]

    if state != None:
        temptbl = temptbl[temptbl.State == state]

    temptbl = temptbl[temptbl.seat == "federal:" + str(e_type)]

    return temptbl

"""
strip_func()
    Helper function to put geoid in same format as other things..namely without
    state and addl 0's.
"""
def strip_func(s):
    return str(s)[2:].lstrip('0')


"""
add_cf_scores()

    Function that adds cf-scores for each precinct + year combo.
        @param: prec_cd_df, cand_df, prec_df, year
        NOTE: assumes all files are of the same year

        @ret: outfile
        # geoid,district_num,candname,cf_score,prop,party,cf_prop,num_votes
        NOTE: each prec-cand combo, has to be converted again for mh_code
"""
def add_cf_scores(prec_cd_df, cand_df, prec_df, state, year, csv_dir):
    colname = 'congdist'

    # only want subset of dime data
    cand_df = pd.DataFrame({
        colname: cand_df['District'],
        'candname': cand_df['Name'],
        'cf_score': cand_df['cfscore'],
        'party': cand_df['Party']})
    if (cand_df[colname]).dtype != int:
        cand_df[colname] = cand_df[colname].map(lambda x: int(x[2:]))
    cand_df.to_csv('%s/%s_cand_%s.csv' % (csv_dir, state, year),
            index=False)

    # prec_cd_df: | cd## | geoid |
    # cand_df: | ... | candname | cd10 | cfscore | ... |
    out_df = pd.merge(prec_cd_df, cand_df, how='inner', on=colname)

    # merging with third file: prec_df
    prec_df['geoid'] = prec_df['geoid'].astype(str)
    out_df = pd.merge(out_df, prec_df, how='left', on='geoid')

    rep_col = 't_USH_R_%s' % year
    dem_col = 't_USH_D_%s' % year
    nv_series = []
    for i, row in out_df.iterrows():
        if row['party'] == 200:
            nv_series.append(row[rep_col])
        elif row['party'] == 100:
            nv_series.append(row[dem_col])
        else:
            assert(False)

    # XXX used in preliminary stuff..still want?
    # prop_series = nv_series / (out_df[rep_col] + out_df[dem_col])

    ret_df = pd.DataFrame({
        'geoid': out_df['geoid'],
        'district_num': out_df[colname],
        'candname': out_df['candname'],
        'cf_score': out_df['cf_score'],
        'party': out_df['party'],
        'num_votes': nv_series})
    ret_df.to_csv('%s/%s_house_%s_final.csv' % (csv_dir, state, year),
            index=False)


"""
prec_to_cd_bycoord()

    Function that takes in the election year (cong hardcorded) and the proper
    longitude and latitude of the center of the precinct.
        @param note: cannot include + sign

    @return: Congressional District that includes the geographical location

    XXX assuming that the geographical point must be in the proper congressional
    district AND that each precinct is only in exactly one CD...
"""
def prec_to_cd_bycoord(year, lon, lat):
    filename = ''
    if year == 2006:
        filename = 'cd_shapefiles/districts110/districts110.shp'
    elif year == 2008:
        filename = 'cd_shapefiles/districts111/districts111.shp'
    elif year == 2010:
        filename = 'cd_shapefiles/districts112/districts112.shp'
    else:
        print 'Error. Please give different election year.'
        return

    dataset = ogr.Open(filename)
    layer = dataset.GetLayerByIndex(0)
    layer.ResetReading()

    # Location for New Orleans: 29.98 N (long), -90.25 E (lat)
    point = ogr.CreateGeometryFromWkt("POINT(" + lon + " " + lat + ")")

    # Transform the point into the specified coordinate system from WGS84
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromEPSG(4326)
    coordTransform = osr.CoordinateTransformation(
            spatialRef, layer.GetSpatialRef())

    point.Transform(coordTransform)

    for feature in layer:
        if feature.GetGeometryRef().Contains(point):
            break

    # @return congressional district
    # based on docs @ http://cdmaps.polisci.ucla.edu/metadata.txt
    return feature.GetFieldAsInteger(2)




"""
prec_scrape()

    Scrapes all voting tabulation districts (by GEOID) from
    http://tigerweb.geo.census.gov/tigerwebmain/Files/tigerweb_tab10_vtd_2000_tx.html
    and outputs 3 csv files (one per election) with:

        geoid | lat | lon | cd

    @return note: to be used with code in data_code/ directory that holds
    information about candidates for specific congressional districts.

"""
def prec_scrape(state, csvdir):
    url = "http://tigerweb.geo.census.gov/tigerwebmain/Files/tigerweb_tab10_vtd_2010_" + state + ".html"
    xml = requests.get(url).text

    geoid_lst = []
    cong110_lst = []
    cong111_lst = []
    cong112_lst = []

    hit_first = False
    for row in web.Element(xml).by_tag('tr'):
        if not hit_first:
            assert(row.by_tag('th')[2].content == 'GEOID')
            assert(row.by_tag('th')[15].content == 'CENTLAT')
            assert(row.by_tag('th')[16].content == 'CENTLON')
            hit_first = True
            continue

        # XXX for tx: geoid_val = row.by_tag('td')[2].content[2:].lstrip('0')
        # XXX for ny: <COUNTY><VTD 4 digits guaranteed>
        county = row.by_tag('td')[4].content.lstrip('0')
        vtd = row.by_tag('td')[5].content.zfill(4)
        geoid_val = county + vtd
        geoid_lst.append(geoid_val)

        lat = row.by_tag('td')[15].content.strip('+')
        lon = row.by_tag('td')[16].content.strip('+')

        cong110_lst.append(prec_to_cd(2006, lon, lat))
        cong111_lst.append(prec_to_cd(2008, lon, lat))
        cong112_lst.append(prec_to_cd(2010, lon, lat))

    df = pd.DataFrame({
        'geoid': geoid_lst,
        'cd06': cong110_lst,
        'cd08': cong111_lst,
        'cd10': cong112_lst})

    df.to_csv('%s/%s_prec_cd.csv' % (csvdir, state), index=False)
    return outfile

"""
prec_to_cd_byshp()

    Determines which precincts are located in which congressional districts by
    overlapping shapefiles.

"""
def precs_to_cd_byshp(state, year, prec_shp_file, csvdir):
    filename = ''
    if year == '2006':
        filename = 'cd_shapefiles/districts110/districts110.shp'
    elif year == '2008':
        filename = 'cd_shapefiles/districts111/districts111.shp'
    elif year == '2010':
        filename = 'cd_shapefiles/districts112/districts112.shp'
    else:
        print 'Error. Please give different election year.'
        return

    # opening file to write out to
    outfile_name = '%s/%s_%s_prec_cd.csv' % (csvdir, state, year)
    outfile = open(outfile_name, 'w+')
    csvwriter = csv.writer(outfile)
    csvwriter.writerow(['geoid', 'congdist'])

    cd_dataset = ogr.Open(filename)
    cd_layer = cd_dataset.GetLayerByIndex(0)
    cd_layer.ResetReading()

    prec_dataset = ogr.Open(prec_shp_file)
    print prec_dataset
    prec_layer = prec_dataset.GetLayerByIndex(0)
    prec_layer.ResetReading()

    prec_spatial_ref = prec_layer.GetSpatialRef()
    cd_spatial_ref = cd_layer.GetSpatialRef()
    coordTransform = osr.CoordinateTransformation(prec_spatial_ref,
            cd_spatial_ref)

    count = 0
    for prec in prec_layer:
        # NOTE: for texas: geoid @ index 0
        geoid = prec.GetFieldAsString(0)

        """
        # for ny: geoid @ index 1(cnty) and 2(vtd pad with 4) combined..
        county = prec.GetFieldAsString(1).lstrip('0')
        vtd = prec.GetFieldAsString(2).zfill(4)
        geoid = county + vtd
        """

        """
        for i in xrange(prec.GetFieldCount()):
            print prec.GetFieldAsString(i)
        break
        """

        # because i can't seem to find by entire precinct area, going to just
        # use coordinates...
        # centroid = prec.GetGeometryRef().Centroid()
        # print centroid

        prec_geo = prec.GetGeometryRef()
        prec_geo.Transform(coordTransform)

        not_found = True

        for cd in cd_layer:
            if cd.GetGeometryRef().Contains(prec_geo) or \
                    cd.GetGeometryRef().Crosses(prec_geo) or \
                    cd.GetGeometryRef().Overlaps(prec_geo) or \
                    cd.GetGeometryRef().Intersect(prec_geo) or \
                    cd.GetGeometryRef().Touches(prec_geo):
                csvwriter.writerow([geoid, cd.GetFieldAsInteger(2)])
                not_found = False
                break

            # found the corresponding cd
            # based on docs @ http://cdmaps.polisci.ucla.edu/metadata.txt

        cd_layer.ResetReading()
        if not_found:
            count += 1


    print 'Num. Skipped: ' + str(count)
    outfile.close()
    return outfile_name


def prec_cd_main(state, year):
    csv_dir = '../data/%s_data/data/%s_%s' % (state, state, year)
    state_data_dir = '../data/%s_data' % state

    # reading in dataframes for speed
    outfile = '%s/%s_%s_prec_cd.csv' % (csv_dir, state, year)
    if not os.path.isfile(outfile):
        # XXX hard-coded ):
        if state == 'tx':
            precs_to_cd_byshp(state, year,
                    '%s/shapefile/tx_shapefile/Texas_VTD.shp' % state_data_dir,
                    csv_dir)
        elif state == 'ny':
            print 'blargl need to add'
            return
        else:
            print 'blargl, no can do'
            return

    prec_cd_df = pd.read_csv(outfile)
    # checking the type to be same as prec_df
    prec_cd_df['geoid'] = prec_cd_df['geoid'].map(str)

    # | geoid | t_USR_R_<yr> | t_USR_D_<yr> |
    prec_df = pd.read_csv('%s/data/orig_data/%s_precvote.csv' % \
            (state_data_dir, state))

    # updating prec_df to have correct geoids
    # XXX new york only?
    # prec_df['geoid'] = prec_df['geoid'].map(strip_func)

    # XXX assuming all for house
    # cand_df := dime data subset that was selected by hand..
    # cand_df = dime_subset(dime_df, state, yr, 'house')
    cand_df = pd.read_csv('%s/data/orig_data/%s_%s_cand_byhand.csv' % \
            (state_data_dir, state, year))

    add_cf_scores(prec_cd_df, cand_df, prec_df, state, year, csv_dir)


def main():
    state = raw_input('State: ')
    year = raw_input('Year: ')

    prec_cd_main(state, year)


if __name__ == "__main__":
    main()

