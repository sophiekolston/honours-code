# %%
# external modules
import geopandas as gpd
import pandas as pd
import importlib
import numpy as np
import time
from collections import defaultdict
import random
from datetime import datetime
from collections import deque
from memory_profiler import profile
import os

# my modules
import erosion_model as erosion
import parse_data as parse
import spatial_ops as spatial
import housing_ops as housing
import visualisation as vis
import utilities as utils
importlib.reload(erosion)
importlib.reload(parse)
importlib.reload(spatial)
importlib.reload(housing)
importlib.reload(vis)
importlib.reload(utils)

# run memory profiler
# mprof run main.py, mprof plot
#@profile
pd.set_option('display.float_format', lambda x: '%.3f' % x)


"""
Main model class

instantiation loads data and prepares the main arrays
model is progressed by calling the step function
model output can be saved to pickle files
parameters are changed during instantiation
    there is a very long list of parameters, see example usage in __main__
"""
class ABM:
    # initialise model (load data, store and calculate variables)
    def __init__(self, min_house_area, cutoffs, monthly_inflation_rate, monthly_erosion_dist,
                 monthly_erosion_moe, erosion_dist_precision, neighbourhood_dist, cliff_erosion_res,
                 monthly_storm_probability, n_slips, n_slips_moe, slip_size, slip_size_moe,
                 neighbour_history_length, n_potential_buyers, afford_multiplier,
                 use_household_income, riskcat_pricereductions_perc, risktolerance_precision,
                 parsed_data_dir, inaccess_deduction):

        # define data passed through
        self.risk_cutoffs = cutoffs
        self.monthly_inflation = monthly_inflation_rate
        self.erosion_res = cliff_erosion_res
        self.n_slips = n_slips
        self.slip_size = slip_size
        self.n_slips_moe = n_slips_moe
        self.slip_size_moe = slip_size_moe
        self.monthly_erosion_dist = monthly_erosion_dist
        self.monthly_erosion_moe = monthly_erosion_moe
        self.erosion_dist_precision = erosion_dist_precision
        self.monthly_storm_probability = monthly_storm_probability
        self.riskcat_pricereds = riskcat_pricereductions_perc
        self.rt_precision = risktolerance_precision
        self.data_folder = parsed_data_dir
        self.inaccessible_deduction = inaccess_deduction
        self.neighbourhood_dist = neighbourhood_dist

        # create potential buyers
        self.buyers = housing.get_potential_buyers(n=n_potential_buyers,
                                                   affordability_multiplier=afford_multiplier,
                                                   income_by_household=use_household_income,
                                                   rt_precision=self.rt_precision,
                                                   csv_path=f'{parsed_data_dir}/ird_wages_2022.csv')

        # load prepped data
        with utils.Timer('data loading'):
            self.homes, self.cliff_homes, self.coastline, cur_cliffs, cur_land, self.parcels, self.neighbours, \
                self.cliff_constraint = self.load_data(neighbourhood_dist=self.neighbourhood_dist,
                                                    value_attr='CV', filter_size=min_house_area)

        # define containers/temporary vars
        self.step_id = 0
        self.year = 1
        self.homes_archive = []
        self.neighbours_archive = deque(maxlen=neighbour_history_length)
        self.cur_homes = self.homes
        self.vis_vars = defaultdict(list)
        self.inaccessible_indices = []
        self.storms = defaultdict(list)
        # there is both land_gdf and land_geom because the code mainly uses land_gdf
        #   for spatial joining. saving seperately means I can use geom for visualisation/saving
        #   without more conversions. cliffs is only cliffs because I only use geometry
        self.geog_archive = {'cliffs': [cur_cliffs], 'land_gdf': [cur_land],
                             'land_geom': [cur_land.geometry[0]], 'step': [0]}
        self.reset_curprev()


    # load data required for model running
    def load_data(self, neighbourhood_dist, calculate_cliffs=False,
                  value_attr='CV', filter_size=None, dem_path=None):
        coastline, cliffs, extent, buildings, parcels = parse.load_prepped_devdata(
            folder=self.data_folder,
            sales_data_type='council')

        if filter_size != None:
            buildings = buildings[buildings.geometry.area >= filter_size]

        if calculate_cliffs:
            cliffs = spatial.define_cliffs(dem_path, coastline, transect_length=50,
                                            transect_seg_length=1, cliff_slope_threshold=60,
                                            filter_dist=True, filter_dist_cutoff=50)

        # create an area of extent that covers where cliffs can
        #   erode (and therefore where cliff proximity is relevent)
        cliff_constraint_data = gpd.read_file(f'{self.data_folder}/cliff_region.gpkg', crs=2193)
        cliff_constraint = gpd.GeoDataFrame(geometry=[cliff_constraint_data.unary_union.buffer(5).simplify(1)])

        # fixing geometry - this should be done in parse, need to fix in future
        buildings = buildings[buildings.is_valid]
        buildings = buildings[buildings.within(extent.geometry.iloc[0])]

        # construct main arrays
        price = np.array(buildings[value_attr], dtype=np.float32)
        geom_centroid = np.array(buildings.centroid)
        geom_building = np.array(buildings.geometry)
        accessible = np.array(np.repeat(True, len(price)), dtype=np.bool)  # will be numbers eventually, just testing
        sticker = np.array(np.repeat('None', len(price)))  # again, numbers eventually
        dists = np.zeros(len(price))
        building_parcel_id = np.array(buildings['id'])
        econ_risk = np.array(np.repeat(0, len(price)))
        on_market = np.array(np.repeat(False, len(price)))
        owner_rt = housing.create_risk_tolerances(len(price), self.rt_precision)
        occupant_number = np.array(np.repeat(0, len(price)))

        # cliff area detection
        in_cliff_area = spatial.x_in_poly(gpd.GeoDataFrame(geometry=geom_building), cliff_constraint)['index_right']
        cliff_area = np.array(np.repeat(False, len(price)))
        cliff_area[in_cliff_area] = True

        # construct supplementary arrays

        # array of neighbours based on spatial matrix
        arr_neighbours = spatial.get_neighbours(buildings.centroid, neighbourhood_dist)
        neighbourval_placeholder = np.array(np.repeat(0, len(price)))

        # construct parcel array
        # filter performed here instead of in dataprep so I can plot it all together later if I want
        parcels_filt = parcels[parcels['id'].isin(buildings['id'])]
        parcel_id = np.array(parcels_filt['id'])
        geom_parcel = np.array(parcels_filt.geometry)
        landarea_parcel = np.array(parcels_filt.geometry.area)
        parcels_df = pd.DataFrame({'id': parcel_id, 'land_area':landarea_parcel, 'geometry':geom_parcel})


        # construct main df
        homes_df = pd.DataFrame({
            'original_value': price,
            'purchase_price': price,
            'value': price,
            'owner_risktolerance': owner_rt,
            'in_cliff_area': cliff_area,
            'dist_to_cliff': dists,
            'accessible': accessible,
            'risk_category': sticker,
            'economic_risk': econ_risk,
            'owner_number': occupant_number,
            'for_sale': on_market,
            'average_neighbour_value': neighbourval_placeholder,
            'parcel_id': building_parcel_id,
            'geom_centroid': geom_centroid,
            'geom_building': geom_building,
            })

        homes_gdf = gpd.GeoDataFrame(homes_df, geometry='geom_building', crs=2193)

        home_incliff_series = gpd.GeoSeries(homes_gdf[homes_gdf['in_cliff_area'] == True].geometry, crs=2193)

        # initial distance calculation
        dists_calced = spatial.get_dist(
                x_geom=homes_gdf,
                geom_incliffarea=home_incliff_series,
                y_geom=cliffs.unary_union,
                access=accessible,
                constrain_extent=cliff_constraint,
                constrain=True,
                simplify_res=self.erosion_res,
                round_dp=3)

        homes_gdf['dist_to_cliff'] = dists_calced

        # calculate average value of neighbours for each home
        homes_gdf['average_neighbour_value'] = housing.get_avg_neighbourval(homes_df, arr_neighbours)

        # reduce memory usage by changing data types. seperate function instead of
        #   during creation to make it easier to change
        homes_gdf = parse.reduce_gdf_memory(homes_gdf)

        # temporary
        #cliff_constraint = cliff_constraint.iloc[0].geometry

        # save final geometry to join back later (useful for saving space in archives)
        homes_geom_name = self.data_folder + f'homes_geom_filt{filter_size}.gpkg'
        if not os.path.isfile(homes_geom_name):
            homes_geom_gdf = homes_gdf.reset_index()  # only need index and geometry
            homes_geom_gdf['geometry'] = homes_geom_gdf.geometry
            homes_geom_gdf = gpd.GeoDataFrame(homes_geom_gdf[['index', 'geometry']], crs=2193)
            homes_geom_gdf.to_file(homes_geom_name, crs=2193)
        
        # return everything
        return homes_gdf, home_incliff_series, coastline, cliffs.unary_union, extent, parcels_df, arr_neighbours, cliff_constraint


    # function that is called at end of each step, reset variables that represent current and previous homes
    def reset_curprev(self):
        # I think the bug is here
        self.prev_homes = self.cur_homes.copy()
        # just copying self.homes because otherwise I would have to go through and create the boolean arrays again
        self.cur_homes = self.cur_homes.copy()  # was self.homes.copy()


    # progress model by 1 step (1 month)
    def step(self):
        """
        Note that currently I have most timing code commented, not the main focus atm

        Order of operations:
        1. calculate erosion rate
        2. determine if a storm is going to occur
        3. erode cliff based on 2 and 3
        4. define if homes are inaccessible based on 3
        5. calculate distance of homes to cliff based on 3
        6. calculate risk category each home is in based on 5
        7. reduce home prices based on 6
        8. get price change of neighbours of each home then alter price of home. based on 7
        9. calculate economic risk based on home values from 7 and 8
        10. sell houses based on 9
        11. return to 1
        """

        self.step_id += 1

        # calculate erosion distance
        #   distance (m) that the entire cliff receedes by
        #with utils.Timer('erosion rate calculation'):
        cur_erosion_rate = erosion.get_erosion_dist(
            monthly_rate=self.monthly_erosion_dist,
            monthly_error=self.monthly_erosion_moe,
            precision=self.erosion_dist_precision)
        print(f'Eroding all cliffs by {cur_erosion_rate} m ({round(cur_erosion_rate * 100, self.erosion_dist_precision)} cm)')

        # detect storm
        #   chance that this step is a stormy month
        #with utils.Timer('storm detection'):
        storm_month = erosion.detect_storm(
            monthly_probability=self.monthly_storm_probability,
            print_if_storm=True)

        # erode cliff
        #   entire cliff receedes by a set amount (with margin of error) as well as
        #   creates slip events (more extreme erosion) if this step is a storm month
        #with utils.Timer('erosion'):
        cur_land, cur_cliff, storm_magnitude, slip_sizes = erosion.erode_by_buffer(
            land=self.geog_archive['land_gdf'][-1],
            cliffs=self.geog_archive['cliffs'][-1],
            dist=cur_erosion_rate,
            simplify_tolerance=self.erosion_res,
            constrain_erosion=True,
            storm=storm_month,
            n_slips=self.n_slips,
            n_slips_moe=self.n_slips_moe,
            slip_size=self.slip_size,
            slip_size_moe=self.slip_size_moe,
            constrain_extent=self.cliff_constraint,
            extreme_buf_res=3,
            land_buf_dist=0.001)

        # save storm information if this step has one
        if storm_month:
            self.storms['step'].append(self.step_id)
            self.storms['magnitude'].append(storm_magnitude)
            self.storms['n_slips'].append(len(slip_sizes))
            self.storms['avg_slip_size'].append(np.mean(slip_sizes))

        # add land and cliffs to list of all for later visualisation
        self.geog_archive['land_gdf'].append(cur_land)
        self.geog_archive['cliffs'].append(cur_cliff)
        self.geog_archive['step'].append(self.step_id)
        self.geog_archive['land_geom'].append(cur_land.geometry[0])

        # find if buildings are no longer accessible (cut off from land/fallen into ocean)
        #with utils.Timer('accessibility check'):
        self.cur_homes['accessible'] = housing.apply_accessible(
            geometries=gpd.GeoDataFrame(geometry=self.prev_homes['geom_building']),
            land=gpd.GeoDataFrame(geometry=self.geog_archive['land_gdf'][-1]),
            accessible_arr=self.prev_homes['accessible'])

        # remove inaccessible neighbours from processsing, and reduce neighbouring home values
        inaccess_val_adjustments, self.neighbours, self.inaccessible_indices = housing.handle_inaccessibility(
            main_df=self.cur_homes,
            neighbour_arr=self.neighbours,
            inaccess_indices=self.inaccessible_indices,
            inaccess_modifier=self.inaccessible_deduction)
        self.cur_homes['value'] *= inaccess_val_adjustments

        # calculate distances to cliff face (where relevant)
        #with utils.Timer('distance calculation'):
        self.cur_homes['dist_to_cliff'] = spatial.get_dist(
            x_geom=self.prev_homes,
            geom_incliffarea=self.cliff_homes,
            y_geom=self.geog_archive['cliffs'][-1],
            access=self.prev_homes['accessible'],
            constrain_extent=self.cliff_constraint,
            constrain=True,
            simplify_res=self.erosion_res,
            round_dp=3)

        # risk categorisation
        #with utils.Timer('risk categorisation and value updates'):
        # first apply risk categories based on distance to cliff edge
        self.cur_homes['risk_category'] = housing.apply_risk_cutoffs(
            dist_array=self.cur_homes['dist_to_cliff'],#self.prev_homes['dist_to_cliff'],
            cutoffs=self.risk_cutoffs,
            accessible_arr=self.cur_homes['accessible'])#self.prev_homes['accessible'])

        # then update value of homes if the risk category changes
        self.cur_homes['value'] -= housing.price_by_riskcatchange(
            homes=self.cur_homes,#model.prev_homes,
            risk=self.cur_homes['risk_category'],#model.prev_homes['risk_category'],
            prev_risk=self.prev_homes['risk_category'],#model.homes_archive[-1]['risk_category'],
            debug=True,
            perc_reductions=self.riskcat_pricereds)

        # add inflation to home values
        #with utils.Timer('inflation calculation'):

        # not using category changed prices
        self.cur_homes['value'] += housing.inflate_prices(
            price_array=self.cur_homes['value'],
            percentage=self.monthly_inflation)
        #self.cur_homes.loc[cur_accessible_homes.index] = cur_accessible_homes

        # check neighbours
        #with utils.Timer('neighbourhood data search'):
        if self.neighbourhood_dist > 0:
            self.cur_homes['average_neighbour_value'] = housing.get_avg_neighbourval(self.cur_homes, self.neighbours)
            self.cur_homes['value'] += housing.modify_neighbour_vals(self.cur_homes, self.prev_homes)

        # calculate economic risk
        #with utils.Timer('economic risk updates'):
        # economic risk is the proportion between the purchase value and current value
        self.cur_homes['economic_risk'] = housing.calc_econ_risk(
            homes=self.cur_homes)#self.prev_homes)

        # attempt to sell homes where relevent
        #with utils.Timer('house sales'):
        # get all homes that attempt or successfully sell
        marketed_homes = housing.sales(
            homes=self.cur_homes,#self.prev_homes,
            buyers=self.buyers,
            attractiveness_proportion=0.01,
            attractiveness_moe=0.005,
            attractiveness_precision=4,
            buyer_sorting='highest',  # random or highest
            notsold_pricered=0.05,
            debug=True)

        # if there are homes in the market
        if len(marketed_homes) > 0:
            # add new buyer information and/or new values for unsold homes
            self.cur_homes.update(marketed_homes, overwrite=True)

        self.cur_homes.loc[self.inaccessible_indices, 'value'] = 0

        #print('Average home value: $', np.mean(self.cur_homes['value']))
        print(f'Change in average home value since last step: ${np.mean(self.cur_homes["value"]) - np.mean(self.prev_homes["value"])}')

        # archive
        #   appending to a list then concatenating later is much faster than concat each time
        #with utils.Timer('archive'):
        self.homes_archive.append(parse.create_archive_df(
            cur_gdf=self.cur_homes,
            step_id=self.step_id,
            year=self.year,
            storm_month=storm_month))

        # new year if 12 steps (months) have passed
        if self.step_id % 12 == 0:
            self.year += 1

        # save info for plotting later
        #riskcat_counts = self.cur_homes.value_counts(['risk_category'])
        #n_inaccessible = len(self.cur_homes['accessible']) - sum(self.cur_homes['accessible'])
        self.vis_vars['n_homes_sold'].append(len(marketed_homes))

        # reset
        self.reset_curprev()


    # save archive of homes, geography and storms to set output folder
    def save_model_data(self, out_dir=''):
        home_archive_df = pd.concat(self.homes_archive, ignore_index=True)
        geog_archive_df = pd.DataFrame(self.geog_archive)
        storm_archive_df = pd.DataFrame(self.storms)


        # create unique(ish) identifier for model run
        dt = datetime.now().strftime('%d%m%Y-%H%M')
        #run_details = f'{self.monthly_erosion_dist}m_{self.monthly_storm_probability}pc_{self.step_id}stp'
        run_details = f'{self.monthly_erosion_dist}m_{self.monthly_storm_probability}pc_{self.neighbourhood_dist}nd_{self.step_id}stp'
        run_dir = f'{out_dir}/{dt}_{run_details}/'

        # create folder in this id name
        os.mkdir(run_dir)

        # save files as pickle
        home_archive_df.to_pickle(run_dir + 'homes.pkl')
        geog_archive_df.to_pickle(run_dir + 'geography.pkl')
        storm_archive_df.to_pickle(run_dir + 'storms.pkl')

        return run_dir

