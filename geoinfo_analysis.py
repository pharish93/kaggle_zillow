import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap, cm

sns.set(font_scale=1.5)

citydict = {}
citydict['Los Angle'] = [34.088537, -118.249923]
citydict['Anaheim'] = [33.838199, -117.924770]
citydict['Irvine'] = [33.683549, -117.793723]
citydict['Long Beach'] = [33.778341, -118.285261]
citydict['Oxnard'] = [34.171196, -119.165045]
citydict['Ventura'] = [34.283106, -119.225597]
citydict['Palmdale'] = [34.612009, -118.127173]
citydict['Lancaster'] = [34.719710, -118.135903]
citydict['Hesperia'] = [34.420196, -117.289121]
citydict['Riverside'] = [33.972528, -117.405517]

def create_basemap( llcrnrlat=20,urcrnrlat=50,llcrnrlon=-130,urcrnrlon=-60, figsize=(16,9) ):
    fig=plt.figure( figsize = figsize )
    Bm = Basemap( projection='merc',
                llcrnrlat=llcrnrlat,urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon,urcrnrlon=urcrnrlon,
                lat_ts=20,resolution='i' )
    # draw coastlines, state and country boundaries, edge of map.
    Bm.drawcoastlines(); Bm.drawstates(); Bm.drawcountries()
    return Bm, fig

def plot_maincities( Bm, citydict ):
    for key, values in citydict.items():
        x , y = Bm( values[1], values[0] )
        Bm.plot( x, y, 'bo', markersize = 5)
        plt.text( x+3000, y+3000, key )


def view_missing(df, target, see_known=True, ignorefirst=False):
    Bm, fig = create_basemap(**CAparms)

    # plot the known data
    if see_known:
        notmiss_df = df.loc[df[target].notnull()]
        groupby = notmiss_df.groupby(target)
        groups = [groupby.get_group(g) for g in groupby.groups]
        groups = groups[1:] if ignorefirst else groups
        print('num groups:  ', len(groups))
        for group in groups:
            x, y = Bm(group['longitude'].values, group['latitude'].values)
            Bm.scatter(x, y, marker='D', s=1)

    # plot the missing data
    missing_target = df[target].isnull()
    if missing_target.any():
        print('{} missing value at column: {}'.format(missing_target.sum(), target))
        missing = df.loc[missing_target, ['latitude', 'longitude']]
        x, y = Bm(missing['longitude'].values, missing['latitude'].values)
        Bm.scatter(x, y, marker='D', s=3, color='yellow', alpha=0.1)
    else:
        print('zero missing value at column: ', target)

    Bm.drawcounties(color='b', linewidth=0.3)

    plot_maincities(Bm, citydict)

    plt.show()

def geo_analysis(df_train,df_test):
    geocolumns = ['latitude', 'longitude'
        , 'propertycountylandusecode', 'propertylandusetypeid', 'propertyzoningdesc'
        , 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip'
        , 'censustractandblock', 'rawcensustractandblock']

    geoprop = df_train[geocolumns]
    # geoprop_test = df_train[geocolumns]


    # let's clean the row without the latitude and longitude value
    geoprop.dropna(axis=0, subset=['latitude', 'longitude'], inplace=True)

    geoprop.loc[:, 'latitude'] = geoprop.loc[:, 'latitude'] / 1e6
    geoprop.loc[:, 'longitude'] = geoprop.loc[:, 'longitude'] / 1e6

    maxlat = (geoprop['latitude']).max()
    maxlon = (geoprop['longitude']).max()
    minlat = (geoprop['latitude']).min()
    minlon = (geoprop['longitude']).min()
    print('maxlat {} minlat {} maxlon {} minlon {}'.format(maxlat, minlat, maxlon, minlon))

    CAparms = {'llcrnrlat': minlat,
               'urcrnrlat': maxlat + 0.2,
               'llcrnrlon': maxlon - 2.5,
               'urcrnrlon': minlon + 2.5}

    Bm, fig = create_basemap()
    x, y = Bm(geoprop['longitude'].values, geoprop['latitude'].values)
    Bm.scatter(x, y, marker='D', color='m', s=1)
    plt.show()

    Bm, fig = create_basemap(**CAparms)
    x, y = Bm(geoprop['longitude'].values, geoprop['latitude'].values)

    Bm.scatter(x, y, marker='D', color='m', s=1)
    Bm.drawcounties(color='b')

    plot_maincities(Bm, citydict)

    plt.show()


    return df_train, df_test