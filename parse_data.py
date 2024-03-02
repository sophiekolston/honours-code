"""
PARSE DATA - load and save data

Loads input data for ABM (census units, homes and values, coastlines, cliffs).
    This data can be exchanged pretty easily, see project readme
Contains functions for saving step data from ABM
"""

import shapely.geometry as shgeo
import geopandas as gpd
import pandas as pd
import os
import copy
import numpy as np
from collections import defaultdict


# load and prepare data 
def dev_dataprep(coastline_file, sales_file, houses_file, buildings_file, cliffs_file, parcels_file,
                 out_crs, house_data_type, bbox=None, land_file=None, default_buildinggeom='building_geometry',
                 building_multipart=True, save_data=True, saved_loc='data/parsed/',
                 centroid_attr='sale_geometry'):
    
    if bbox != None:
        # get extent polygon for clipping to AOI
        # copy and paste from bbox finder
        # eg. bbox=[1757353.3212, 5921660.2905, 1762371.7687, 5927057.1612]
        extent_poly = shgeo.Polygon([bbox[2:], (bbox[:2][0], bbox[2:][1]), bbox[:2], (bbox[2:][0], bbox[:2][1])])
    else:
        extent_poly = gpd.read_file(land_file, crs=4326)

    # coastline
    coast_raw = gpd.read_file(coastline_file, crs=4326).to_crs(out_crs)
    coast = gpd.clip(coast_raw, extent_poly)  # coastline in AOI

    if house_data_type == 'corelogic':  # proprietary data available to UoA students
        # sales 
        sales_raw = pd.read_csv(sales_file, encoding='ISO-8859-1', low_memory=False)
        houses_raw = pd.read_csv(houses_file, encoding='ISO-8859-1')
        houses_withgeom = houses_raw.dropna(subset=['CL_Latitude', 'CL_Longitude'], axis=0)
        houses_gdf = gpd.GeoDataFrame(houses_withgeom, 
                                geometry=gpd.points_from_xy(houses_withgeom['CL_Longitude'], houses_withgeom['CL_Latitude']),
                                crs=4326).to_crs(out_crs)
        houses_indev = gpd.clip(houses_gdf, extent_poly)
        housesales_dev = pd.merge(houses_indev, sales_raw, on='CL_QPID')
        housesales_attr = housesales_dev[['CL_Sale_Price_Gross', 'CL_Sale_Date', 'geometry']]

        # buildings
        buildings_raw = gpd.read_file(buildings_file, crs=4326).to_crs(out_crs)
        buildings_raw['building_geometry'] = buildings_raw.geometry.copy()
        housesales_attr[centroid_attr] = housesales_attr.geometry.copy()
        building_sales = gpd.sjoin(housesales_attr, buildings_raw).set_geometry(default_buildinggeom)
        if not building_multipart:
            building_sales = building_sales.explode()
            building_sales['building_geometry'] = building_sales.geometry
        building_sales_attr = building_sales[['CL_Sale_Price_Gross', 'CL_Sale_Date', 'building_geometry', centroid_attr]]
        buildings = building_sales_attr.rename({'CL_Sale_Price_Gross': 'value', 'CL_Sale_Date': 'last_sale_date'}, axis=1)
        
        # get only the first index of each building
        buildings['last_sale_date'] = pd.to_datetime(buildings['last_sale_date'])
        idx = buildings.groupby('id')['last_sale_date'].idxmax()
        buildings = buildings.loc[idx]
    elif house_data_type == 'council':  # proprietary data available only upon request to Auckland council
        values = gpd.read_file(sales_file, crs=2193)

        buildings_raw = gpd.read_file(buildings_file, crs=4326).to_crs(2193)
        buildings_raw['building_geometry'] = buildings_raw.geometry.copy()
        buildings_raw[centroid_attr] = buildings_raw.centroid
        if not building_multipart:
            buildings_raw = buildings_raw.explode()
            buildings_raw['building_geometry'] = buildings_raw.geometry

        building_sales = gpd.sjoin(buildings_raw, values)

        suitable_landuse = ['Single units, excluding bach', 'Multi-unit',
                            'Multi-use within residential', 'Residential - vacant']
        buildings = building_sales[building_sales['LANDUSEDESCRIPTION'].isin(suitable_landuse)]
        buildings = buildings.drop(['index_right'], axis=1)

        buildings = buildings[['CV', 'LV', 'IV', 'building_geometry', centroid_attr]]

    # land parcels
    parcels = gpd.read_file(parcels_file, crs=4326).to_crs(out_crs)
    parcels_clip = parcels.clip(extent_poly)
    parcels_attr = parcels_clip[['id', 'geometry']].rename({'id': 'parcel_id'})
    parcels_clip = parcels_attr.clip(extent_poly)  # make sure parcels and land line up

    # join buildings and parcels
    buildings_wparcels = buildings.set_geometry(centroid_attr).sjoin(parcels_attr, how='left', op='intersects')
    buildings_wparcels = buildings_wparcels.set_geometry(default_buildinggeom)
    
    # cliffs
    cliffs = gpd.read_file(cliffs_file, crs=4326).to_crs(out_crs)

    # save files if requested
    if save_data:
        cliffs.to_file(saved_loc + 'cliffs.gpkg', driver='GPKG')
        coast.to_file(saved_loc + 'coastline.gpkg', driver='GPKG')
        extent_poly.to_file(saved_loc + 'extent.gpkg', driver='GPKG')
        # have to drop geom columns for data type reasons
        buildings_wparcels.drop(columns=[centroid_attr], 
            axis=1).to_file(saved_loc + f'buildings_{house_data_type}.gpkg', 
                            driver='GPKG')
        parcels_clip.to_file(saved_loc + 'parcels.gpkg', driver='GPKG')

    # return all
    return coast, cliffs, extent_poly, buildings_wparcels, parcels_clip


# load data that has already been manipulated to avoid unnecessary processing
def load_prepped_devdata(folder, sales_data_type='council'):
    cliffs = gpd.read_file(folder + 'cliffs.gpkg')
    coast = gpd.read_file(folder + 'coastline.gpkg')
    extent = gpd.read_file(folder + 'extent.gpkg')

    buildings = gpd.read_file(folder + f'buildings_{sales_data_type}.gpkg')
    buildings['centroid'] = buildings.centroid

    parcels = gpd.read_file(folder + 'parcels.gpkg')

    return coast, cliffs, extent, buildings, parcels


# convert gdf to df with set data types to save memory
def reduce_gdf_memory(main_gdf):

    df = copy.deepcopy(main_gdf)

    # fixed, unlikely to go over 127
    df['owner_number'] = df['owner_number'].astype('int8') 

    # value wont go into billions
    df['original_value'] = df['original_value'].astype('float32')  
    df['purchase_price'] = df['purchase_price'].astype('float32')
    df['value'] = df['value'].astype('float32')
    df['average_neighbour_value'] = df['average_neighbour_value'].astype('float32')

    # risk tolerance and economic risk is fixed between 0 and 1
    df['owner_risktolerance'] = df['owner_risktolerance'].astype('float16')
    df['economic_risk'] = df['economic_risk'].astype('float32')
    
    # currently not doing anything with parcel ID
    df = df.drop('parcel_id', axis=1)

    return df


# convert gdf from a step to df to put in archive
#   remove redundant information to save memory
def create_archive_df(cur_gdf, step_id, year, storm_month):

    homes_withstepinfo = cur_gdf.copy() 

    # add step information
    homes_withstepinfo['house_index'], homes_withstepinfo['step'], homes_withstepinfo['year'], \
        homes_withstepinfo['storm_step'] = [cur_gdf.index, step_id, year, storm_month]

    # justification here - don't need to save geometry as this is fixed, faster to join later
    df_archive = pd.DataFrame(homes_withstepinfo)
    df_archive = df_archive.loc[:, ~df_archive.columns.str.contains('geom')]

    # convert to more efficient data types
    df_archive['step'] = df_archive['step'].astype('int32')
    df_archive['year'] = df_archive['year'].astype('int16')
    df_archive['house_index'] = df_archive['house_index'].astype('int16')

    return df_archive


# create summary stats of homes archive dataframe
def summarise_home_archive(archive_df):

    # group by step, summarise in various ways
    # first on all data
    df_grp = archive_df.groupby('step').agg({
        'value': [np.mean, np.median, sum],
        'risk_category': lambda x: x.value_counts().to_dict(),
        'accessible': lambda x: (x == False).sum(),
        'storm_step': 'first',
        'owner_risktolerance': [np.mean, np.std],
        'owner_number': [np.mean, sum]
    }).reset_index()

    # save categorical/value counts as seperate values
    value_counts_df = df_grp['risk_category']['<lambda>'].apply(pd.Series)
    value_counts_df = value_counts_df.rename(columns=lambda x: f'{x}_count')
    result = pd.concat([df_grp.drop('risk_category', axis=1), value_counts_df], axis=1)
    result = result.fillna(0)

    # rename columns
    df_grp = result.rename({
        ('value', 'median'): 'median_value',
        ('value', 'mean'): 'mean_value',
        ('value', 'sum'): 'total_value',
        ('accessible', '<lambda>'): 'n_inaccessible',
        ('step', ''): 'step',
        ('storm_step', 'first'): 'storm',
        ('owner_risktolerance', 'mean'): 'mean_rt',
        ('owner_risktolerance', 'std'): 'std_rt',
        ('owner_number', 'mean'): 'mean_ownernum',
        ('owner_number', 'sum'): 'total_ownernum'
        }, axis=1)
    
    return df_grp


def stats_by_spatialunit(homes_archive_df, step_a, step_b, extent_gdf, sa1_path, geom_path,
                         vars=['value', 'owner_risktolerance', 'average_neighbour_value']):

    #all_homes = copy.deepcopy(homes_archive_df)
    all_homes = homes_archive_df

    # filter out sa1 stuff
    # whenever geometry is extracted it means there will be a merge or join or something that geopandas
    #   automatically converts back to pandas for some reason, so geometry needs to be in a column to 
    #   convert back
    sa1 = gpd.read_file(sa1_path, crs=2193)
    sa1_clip = sa1.loc[sa1['LANDWATER_NAME'] == 'Mainland'].clip(extent_gdf)
    sa1_clip['geometry'] = sa1_clip.geometry
    sa1_clip = sa1_clip[['SA12018_V1_00', 'geometry']]
    areas = gpd.GeoDataFrame(sa1_clip)

    # get dfs at required steps
    homes_first = all_homes.loc[all_homes['step'] == step_a][vars]
    homes_last = all_homes.loc[all_homes['step'] == step_b][vars]

    # calculate percentage difference
    container = {}
    for var in vars:
        if var == 'owner_risktolerance':
            var_change = ((homes_last[var].values - homes_first[var].values) / homes_last[var].values) * 100
            container[var] = var_change
        else:
            # has to be first to last otherwise divide by 0 on inaccessible homes
            var_perc_change = -((homes_first[var].values - homes_last[var].values) / homes_first[var].values) * 100
            container[var] = var_perc_change
    homes_diff = pd.DataFrame(container)

    # add home geometry
    homes_geom = gpd.read_file(geom_path)
    homes_diff_gdf = homes_geom.merge(homes_diff, right_index=True, left_on='index')
    homes_diff_gdf['centroid'] = homes_diff_gdf.geometry.centroid
    homes_diff_gdf = homes_diff_gdf.set_geometry('centroid')

    # spatial join homes to sa1
    homes_areas = gpd.sjoin(homes_diff_gdf, areas, how='left', predicate='intersects')
    homes_areas = homes_areas[vars + ['SA12018_V1_00']]
    sa1_home_diff = homes_areas.groupby('SA12018_V1_00').mean().reset_index()

    # create gdf
    sa1_home_gdf = gpd.GeoDataFrame(sa1_home_diff.merge(areas))

    # fill na values with -100 for avg neighbour values (when no more neighbours)
    sa1_home_gdf['average_neighbour_value'] = sa1_home_gdf['average_neighbour_value'].fillna(-100)

    return sa1_home_gdf


# iterate over a list of file paths and summarise each home df and save as pickle
#   saves memory in the long run
def summarise_homes_all_runs(run_id_list, input_folder='output_runs/'):
    for run_id in run_id_list:
        df_homes = pd.read_pickle(f'{input_folder}/{run_id}/homes.pkl')
        df_homes_grp = summarise_home_archive(df_homes)
        df_cliff_homes_grp = summarise_home_archive(df_homes.loc[df_homes['in_cliff_area'] == True])

        df_cliff_homes_grp.to_pickle(f'{input_folder}/{run_id}/grouped_cliff_homes.pkl')
        df_homes_grp.to_pickle(f'{input_folder}/{run_id}/grouped_homes.pkl')

        del df_homes


def read_land_geography(run_id):
    df_geog = pd.read_pickle(f'output_runs/{run_id}/geography.pkl')
    gdf_geog = gpd.GeoDataFrame(df_geog, geometry='land_geom')
    return gdf_geog


# load homes created in the above function
def load_all_summarised_runs(run_id_list, group_by, input_folder='output_runs', ):
    container = defaultdict(dict)
    for run_id in run_id_list:
        df_homes_grp = pd.read_pickle(f'{input_folder}/{run_id}/grouped_homes.pkl')
        df_cliff_homes_grp = pd.read_pickle(f'{input_folder}/{run_id}/grouped_cliff_homes.pkl')

        file_split = run_id.split('_')
        erosion_rate = file_split[1].strip('m')
        storm_freq = file_split[2].strip('pc')
        steps = file_split[3].strip('stp')

        if group_by == 'gradual':
            container['all_homes'][erosion_rate] = df_homes_grp
            container['cliff_homes'][erosion_rate] = df_cliff_homes_grp
        else:
            container['all_homes'][storm_freq] = df_homes_grp
            container['cliff_homes'][storm_freq] = df_cliff_homes_grp

    return container


def homes_to_spatialsummary(runs, extent_path='data/input/extent.gpkg', 
                            sa1_path='data/source/sa1/statistical-area-1-2018-generalised.gpkg',
                            home_geom_path='data/homes_geom_filt40.gpkg'):
    extent = gpd.read_file(extent_path)
    
    for run_id in runs:
        df_homes = pd.read_pickle(f'output_runs/{run_id}/homes.pkl')
        gdf = stats_by_spatialunit(df_homes, 1, 2400, extent, sa1_path, home_geom_path)
        gdf.to_file(f'output_runs/{run_id}/home_changes_sa1.gpkg', crs=2193)
        print('processed run:', run_id)