'''
Erosion Model(s)
'''

import numpy as np
import shapely.geometry as shgeo
import geopandas as gpd
import random
import time
import matplotlib.pyplot as plt
import shapely
from shapely.ops import unary_union
import utilities as utils

import warnings
warnings.filterwarnings("ignore")


# calculate erosion distance based on chance, set distance and margin of error
def get_erosion_dist(monthly_rate, monthly_error=0.005, precision=4):
    """
    notes from meeting with Giovanni

    1cm +/- 0.5cm per year
        - constant across all years
    extreme events
        - probability of 1 in 10 yer year that there will be extreme event
        - during event, 5 +/- 2 slips 
        - slips will be 50cm to 2m in size
    all probabilities are uniform distribution (random number)
    """

    error = np.round(np.random.uniform(-monthly_error, monthly_error), precision)
    dist = np.round(monthly_rate + error, precision)

    return dist


# detect storm event (random chance of storm = True)
def detect_storm(monthly_probability, print_if_storm=False):
    rnd = int(np.random.uniform(0, 100))

    storm = False
    if rnd < monthly_probability:
        storm = True
        if print_if_storm:
            print('Storm event!')

    return storm


# calculate the magnitude of a storm, for plotting
#   basic calc, saved as function for easier modification
def get_storm_magnitude(slip_list):
    magnitude = sum(slip_list)
    #magnitude = len(slip_list) * np.mean(slip_list)
    return int(magnitude)


# create geometry of extreme events (slips)
def get_extreme_events(n, n_moe, size, size_moe, cliffs, buf_res=3, slip_dist_precision=2):
    
    slip_sizes = []

    # randomly modify number of slips based on set margin of error
    corrected_n = n + (np.random.randint(-n_moe, n_moe))
    
    # return empty if no slips
    if corrected_n < 1:
        return []
    
    else:
        # create points randomnly, adding them to a list
        slips = []
        for i in range(corrected_n):
            rnd_dist = random.random()  # randint between 0 and 1
            #pt = cliffs.unary_union.interpolate(rnd_dist, normalized=True)  # get point at random position along cliff line
            pt = cliffs.interpolate(rnd_dist, normalized=True)

            # get size of slip
            slip_size = round(size + (np.random.uniform(-size_moe, size_moe)), slip_dist_precision)
            pt_buf = pt.buffer(slip_size, resolution=buf_res)  # buffer by the size of the erosion event
            #print(pt)
            slips.append(pt_buf)  # save all together
            slip_sizes.append(slip_size)

    return slips, slip_sizes


# 'erode' a land polygon 
#   calculate the shape of land after it has been eroded
#   accounts for constant erosion and slip events
def calculate_land_shape(new_land, slips, buffered_cliffs, simplify_tolerance,
                         constrain_extent=None, constrain_erosion=False):
    # calculate difference in land shape based on erosion
    #   this part is done on a cliff-by-cliff basis so that if we decide to use different erosion
    #   rates on different cliffs it will be easier to update. may be performance hit
    all_cliff_removals = []
    combined_geoms = None
    for i, cliff in enumerate(list(buffered_cliffs)):  # for each cliff polygon
        
        # if extreme erosion events are requested
        if len(slips) > 0:
            pts_on_cliff = []  # container for points spatially on current cliff
            for pt in slips:
                if cliff.intersects(pt):  # if the point intersects with the cliff, save it
                    pts_on_cliff.append(pt)
            # combine cliff and extreme points geometry to polygon
            combined_geoms = unary_union([cliff] + pts_on_cliff)
            #combined_geoms = gpd.GeoSeries([cliff] + pts_on_cliff).unary_union
        else:
            combined_geoms = cliff  # just use cliff if no extreme events

        # if the erosion is to be constrained to a set polygon 
        if constrain_erosion:
            # clip the combined geometries to the set extent
            #clipped_geoms = gpd.GeoSeries(combined_geoms).clip(constrain_extent)
            clipped_geoms = combined_geoms.intersection(constrain_extent.geometry[0])
            #print(clipped_geoms)
            geoms = clipped_geoms
            #all_cliff_removals.append(geoms[0])
            all_cliff_removals.append(geoms)
           
        else:
            geoms = combined_geoms
            all_cliff_removals.append(geoms)

        # remove the erosion difference from the land geometry
        dif = new_land.difference(geoms)
        new_land = gpd.GeoDataFrame(geometry=dif)

    # convert to polygon (from multipolygon)
    #   this removes isolated geometry (islands). we assume that islands cut off from the mainland
    #   due to erosion are removed from processing. take the largest size polygon
    # if the land is a multipolygon
    if type(new_land.geometry[0]) == shgeo.MultiPolygon:
        all_geoms = list(*new_land.geometry)  # unpack geometries
        all_geoms_areas = [x.area for x in all_geoms]  # calculate areas
        maxsize_i = (all_geoms_areas.index(max(all_geoms_areas)))  # get the index of the largest polygon
        land_gdf = gpd.GeoDataFrame(geometry=[all_geoms[maxsize_i]])  # save the largest polygon as new gdf
    else:
        land_gdf = new_land  # otherwise, just rename the variable

    land_gdf = land_gdf.simplify(simplify_tolerance)

    return land_gdf, all_cliff_removals


# convert cliff polygon (buffered) to linestring
def cliff_poly_to_line(cliff_removals, land, simplify_tolerance, land_buf_dist):

    # get boundary of cliff removal polygons, convert to gdf
    all_cliff_removals = [x.boundary for x in cliff_removals]
    #cliff_removal_gdf = gpd.GeoDataFrame(geometry=all_cliff_removals)

    # clip poly gdf to area immediately surrounding land

    # use the old land here
    #print(unary_union(all_cliff_removals).intersection(land.geometry[0].buffer(land_buf_dist)))
    cliff_gdf = unary_union(all_cliff_removals).intersection(land.geometry[0].buffer(land_buf_dist))
    #cliff_gdf = gpd.clip(cliff_removal_gdf, gpd.GeoSeries(land.buffer(land_buf_dist)))

    """cliffs_poly = gpd.clip(land_gdf.buffer(land_buf_dist, join_style=3, cap_style=2), gpd.GeoSeries(all_cliff_removals))    
    # next, convert this polygon to a line by getting the boundary of the geometries
    cliff_lines = ([x.boundary for x in list(*cliffs_poly.geometry)]) 
    # lastly, save these lines as a geodataframe
    cliff_gdf = gpd.GeoDataFrame(geometry=cliff_lines) """

    # simplify geometry
    #   with small erosion distances the geometry becomes complex very fast and greatly 
    #   increases processing time. make sure to adjust the simplify tolerance based on the 
    #   resolution of analysis / distance of erosion
    cliff_gdf = cliff_gdf.simplify(tolerance=simplify_tolerance)

    return cliff_gdf


"""
Erode by buffer

represents erosion along a cliff through buffering. the process is as follows:
    1. buffer the input cliff lines by a set erosion rate/distance
    2. randomly create extreme erosion events along the cliff lines
        - these events are represented by buffers along the new cliff line that 
            are larger than the standard erosion rate (set manually)
    3. join the extreme event polygons and cliff line buffers
    4. clip the input land polygon by this joined polygon
    5. move the cliff line to the new position by extracting the boundary of the
        joined polygon, buffering the new land polygon by a tiny amount then 
        clipping the boundary to the land buffer
    6. output are new land and cliff polygons
"""
def erode_by_buffer(land, cliffs, dist, simplify_tolerance=0.1, 
                    n_slips=1, slip_size=30, extreme_buf_res=3,
                    constrain_erosion=True, constrain_extent=None,
                    land_buf_dist=0.001, slip_size_moe=5, 
                    n_slips_moe=1, storm=False):

    if type(cliffs) == gpd.GeoDataFrame:
        cliffs = cliffs.unary_union

    # buffer cliffs by erosion distance
    # join and cap styles here are very important. rounded ones will make the buffer 
    #   intrude on other sections of the cliff line
    buf = cliffs.buffer(dist, single_sided=False, join_style=3, cap_style=1) 
    #buf_geo = gpd.GeoSeries(buf).explode()  # seperate into singlepart geoms
    buf_geo = list(buf) #buf.explode()
    
    # make sure geometry structure is correct
    land_clean = land[~land.geometry.isna()] 
    #clean_polys = buf_geo.buffer(0).unary_union
    clean_polys = unary_union(buf_geo)#.buffer(0)

    # create extreme geometry
    if storm:
        slip_geom, slip_sizes = get_extreme_events(n_slips, n_slips_moe, slip_size, slip_size_moe, 
                                    cliffs, extreme_buf_res)
    else:
        slip_geom = []
        slip_sizes = 0

    # get geometry of eroded land
    final_land, all_cliff_removals = calculate_land_shape(land_clean, slip_geom, clean_polys,
                                simplify_tolerance, constrain_extent, constrain_erosion)

    # find new cliff locations
    final_cliff = cliff_poly_to_line(all_cliff_removals, final_land, simplify_tolerance,
                                    land_buf_dist)

    # if slips, calculate storm magnitude
    if storm:
        storm_magnitude = get_storm_magnitude(slip_sizes)
    else:
        storm_magnitude = 0

    return final_land, final_cliff, storm_magnitude, slip_sizes