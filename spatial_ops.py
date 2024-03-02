import numpy as np
import geopandas as gpd
from scipy.spatial import distance_matrix
import rasterio
from shapely.geometry import Point
from shapely.geometry import LineString
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
import shapely
import importlib
from scipy.spatial.distance import euclidean

import utilities as utils
importlib.reload(utils)


# check if polygons are contained in a multipolygon
#   used for checking if homes are within land
def x_in_poly(multipolygon, polygons):
    # Perform spatial join
    joined_data = gpd.sjoin(polygons, multipolygon, how='left', op='contains')
    
    # Count the number of polygons within the multipolygon
    num_polygons_within = len(joined_data)
    
    return joined_data


# alternative version without spatial joining
def x_in_poly_(multipolygon, polygons):
    # Create the spatial index
    sindex = polygons.sindex
    
    # Create an empty list to store the results
    joined_data = []
    
    # Loop through each polygon in multipolygon
    for idx, poly in multipolygon.iterrows():
        # Get the bounding box coordinates of the polygon
        bounds = poly.geometry.bounds
        
        # Get the indices of polygons that potentially overlap with the bounding box
        possible_matches_index = list(sindex.intersection(bounds))
        possible_matches = polygons.iloc[possible_matches_index]
        
        # Perform the actual spatial join
        precise_matches = possible_matches[possible_matches.overlaps(poly.geometry)]
        
        # Append the results to the list
        joined_data.append(precise_matches)
    
    # Concatenate the results
    joined_data = pd.concat(joined_data)
    
    # Count the number of polygons within the multipolygon
    num_polygons_within = len(joined_data)
    
    return joined_data#, num_polygons_within


# create constraint of cliff locations (only erode in those areas)
#   NOT VERY ACCURATE, is subject to misdirected line geometry. recommend using
#   a manually created constraint layer
def create_cliff_constraint(first_cliff, land, constrain_dist=500):
    buf = gpd.GeoSeries(first_cliff.unary_union).buffer(constrain_dist, join_style=3, cap_style=2,
                             resolution=5)
    buf_clip = buf.clip(land)
    buf_clip.plot()

    # buffer to ensure the boundary of the land is included
    #   otherwise the cliffs will include this boundary and the cliffs will stay as polygons
    buf_extended = buf_clip.buffer(1)  

    return gpd.GeoDataFrame(geometry=buf_extended)  # gdf as it is only used in spatial ops


# get distance between x and y geometries
#   used for distance of homes to cliff, constrained by homes within cliff area to 
#   make processing make more sense and save time
def get_dist(x_geom, geom_incliffarea, y_geom, access, constrain_extent, constrain=True, 
             simplify_res=0.001, round_dp=3):
    
    # don't need to do this every time, indexes won't change
    # gpd.GeoSeries(x_geom[x_geom['in_cliff_area'] == True].geometry, crs=2193)

    if constrain:
        dists = geom_incliffarea.distance(y_geom)

        #with utils.Timer('filtering distances'):
        dists_with_nan = np.full(len(x_geom), np.nan)

        # Set distances to their respective values where x_geom['in_cliff_area'] is True
        dists_with_nan[x_geom['in_cliff_area']] = dists

        # Set distances to nan where the house is no longer accessible
        dists_with_nan[~access] = np.nan

        calced_dists = dists_with_nan

    else:
        calced_dists = geom_incliffarea.distance(y_geom)

    formatted_dists = np.array(calced_dists, dtype=np.float32)
    
    #formatted_dists = np.array(calced_dists, dtype=np.float16)
    return formatted_dists


# find neighbouring properties
#   create a distance matrix of every home geometry, results in a numpy array 
#   of indices for each home that are within a given distance. only needs to be 
#   run once as home geometry is constant
def get_neighbours(x_geom, dist):
    geoms_arr = np.array([(point.x, point.y) for point in x_geom])

    dists = distance_matrix(geoms_arr, geoms_arr)

    neighbours = [np.nonzero(subarr < dist)[0] for subarr in dists]

    return neighbours


# 15/09/23 not used
def get_new_parcels(parcels:gpd.GeoDataFrame, new_land:gpd.GeoDataFrame, 
                    return_both=True):
    # do I need to do both of these operations?
    lost_parcels = parcels.overlay(new_land, how='difference')
    retained_parcels = parcels.clip(new_land)

    if return_both:
        return retained_parcels, lost_parcels
    else:
        return retained_parcels


# sample DEM values (or any raster)
def sample_dem(point, dem, band_object):
        row, col = dem.index(point.x, point.y)
        val = band_object[row, col]
        return val


# calculate slope
def calculate_slope(z1, z2, segment_length):
    rise = abs(z1 - z2)
    run = segment_length
    slope = rise / run
    return slope


# define cliff edges based on elevation changes across each coastline segment
#   for each coast line, generate a perpendicular transect at each line in the 
#   linestring. find the change in slope along this transect. if this change is 
#   over a threshold, define that section of the coast as a cliff
def define_cliffs(dem_path, coastline_gdf, transect_length=50, transect_seg_length=1,
                  cliff_slope_threshold=60, unite=False, save=True, 
                  save_path='data/parsed/', filter_dist=True, filter_dist_cutoff=50):
    
    # create transects along line
    transects = []
    for _, row in coastline_gdf.iterrows():
        if row.geometry.geom_type == 'MultiLineString':
            lines = list(row.geometry)
        else: # assuming the geometry is LineString
            lines = [row.geometry]
        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords)-1):
                x1, y1 = coords[i]
                x2, y2 = coords[i+1]
                azimuth = math.atan2(y2-y1, x2-x1)
                azimuth_perpendicular = azimuth + math.pi / 2
                x3 = x1 + math.cos(azimuth_perpendicular) * transect_length / 2
                y3 = y1 + math.sin(azimuth_perpendicular) * transect_length / 2
                x4 = x1 - math.cos(azimuth_perpendicular) * transect_length / 2
                y4 = y1 - math.sin(azimuth_perpendicular) * transect_length / 2
                transect = LineString([(x3, y3), (x4, y4)])
                transects.append(transect)
    transects_gdf = gpd.GeoDataFrame(geometry=transects, crs=2193)

    # load DEM and band object
    dem = rasterio.open(dem_path)
    band_object = dem.read(1)
    #dem_data = np.where(band_object < 0, np.nan, band_object)

    # compute slope for each segment of each transect
    transects_gdf['is_cliff'] = False
    for idx, transect in transects_gdf.iterrows():
        num_segments = int(transect.geometry.length / transect_seg_length)
        for i in range(num_segments):
            p1 = transect.geometry.interpolate(i * transect_seg_length)
            p2 = transect.geometry.interpolate((i + 1) * transect_seg_length)
            z1 = sample_dem(p1, dem, band_object)
            z2 = sample_dem(p2, dem, band_object)

            # ignore values less than 0 (in ocean)
            z1 = 0 if z1 < 0 else z1
            z2 = 0 if z2 < 0 else z2

            slope = calculate_slope(z1, z2, transect_seg_length)
            slope_degrees = np.arctan(slope) * (180 / np.pi)
            if slope_degrees > cliff_slope_threshold and abs(z1 - z2) > 0.5:
                transects_gdf.loc[idx, 'is_cliff'] = True
                break  # No need to check further if we found a cliff
    
    # convert transects to cliffs
    coastline_ls = coastline_gdf.explode()
    coastline_ls['is_cliff'] = list(transects_gdf['is_cliff'])
    cliffs = coastline_ls[coastline_ls['is_cliff'] == True]
    #non_cliff_coast = coastline_ls[coastline_ls['is_cliff'] == False]

    if filter_dist:
        merged_cliffs = shapely.ops.linemerge([x for x in cliffs.geometry])
        merged_gdf = gpd.GeoDataFrame(geometry=[merged_cliffs]).explode()
        merged_gdf['length'] = merged_gdf.length
        cliffs = merged_gdf.loc[merged_gdf['length'] > filter_dist_cutoff]

    if save:
        cliffs.to_file(save_path + 'cliffs.gpkg', driver='GPKG', crs=2193)
        #non_cliff_coast.to_file(save_path + 'non_cliff_coast.gpkg', driver='GPKG', crs=2193)

    if unite:
        return cliffs.unary_union #non_cliff_coast.unary_union
    else:
        return cliffs #non_cliff_coast
