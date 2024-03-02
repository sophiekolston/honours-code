import numpy as np
import spatial_ops as spatial
import copy
import pandas as pd
from collections import defaultdict


# increase home value by set percentage to represent inflation
def inflate_prices(price_array, percentage):
    #return price_array + (price_array * (percentage / 100))
    return price_array * (percentage / 100)


# assign risk categories based on disance to cliff edge
def apply_risk_cutoffs(dist_array, cutoffs, accessible_arr):

    conditions = [
        # when distance is less than or equal to 0, or greater than the first cutoff, 
        #   or not accessible, set to None
        (dist_array <= 0) | (dist_array > cutoffs[0]) | (accessible_arr == False),
        # when distance is less than or equal to the first cutoff and less than the 
        #   second cutoff, set to Uninsurable
        (dist_array <= cutoffs[0]) & (dist_array > cutoffs[1]),
        # when distance is less than or equal to the second cutoff, and greater than
        #   the third cutoff, set to Yellow
        (dist_array <= cutoffs[1]) & (dist_array > cutoffs[2]),
        # when distance is less than or equal to the third cutoff, and greater than
        #   0, set to Red
        (dist_array <= cutoffs[2]) & (dist_array > 0) 
    ]

    # possible options for the select
    choices = ['None', 'Uninsurable', 'Yellow', 'Red']

    # select from list based on conditions
    result = np.select(conditions, choices, default='None')

    return result


# define if a home is acccessible - whether it has fallen into the ocean or not
#   if the geometry is contained with land or not
def apply_accessible(geometries, land, accessible_arr):
    access = copy.deepcopy(accessible_arr)

    # spatial indexing here
    contained = spatial.x_in_poly(geometries, land)
    
    all_indices = set(np.arange(access.size))
    complement_indices = list(all_indices.difference(contained['index_right']))

    access[complement_indices] = False

    return access


# get trends of 'neighbour' property values
def get_neighbour_trends(neighbours_archive):
    diffs = []

    for i in range(1, len(neighbours_archive)):
        diff = np.subtract(neighbours_archive[i][1], neighbours_archive[i - 1][1])
        diffs.append(diff)

    return diffs


# filter upper band data (for IRD spreadsheet)
def filter_upperband(x, max_limit=300000):
        i = x.replace('$', '').replace(',', '').replace('Over ', '').replace(' ', '').split('-')
        return i[1] if len(i) > 1 else max_limit


# create risk tolerances
# currently uniform/random distribution between 0 and 1
def create_risk_tolerances(n, precision, categorise=False, rt_category_bins=[0, 0.3, 0.6, 1],
                           rt_category_labels=['low', 'medium', 'high']):
    risktols = np.around(np.random.uniform(0, 1, n), precision)

    if categorise:
        risktol_categories = pd.cut(risktols, bins=rt_category_bins, labels=rt_category_labels)
        return risktols, risktol_categories
    else:
        return risktols


# create list of potential buyers
#   based on inland revenue data of wages/income (no difference to them) in 2022
#   get number of individuals in each wage bracket, then take a sample from this distribution
#   result is a distribution of potential buyers that have income close to real data
#   then add risk tolerance, uniform number between 0 and 1 (and categorised for vis)
def get_potential_buyers(n, min_wage=10000, csv_path='data/wealth/ird_wages_2022.csv',
                         group=False, affordability_multiplier=8, 
                         income_by_household=True, rt_precision=3):

    # get wages/income = wealth
    df = pd.read_csv(csv_path)

    df['upper_band'] = pd.to_numeric(df['band'].apply(filter_upperband))  # convert to numeric
    
    # if doing household income, assuming 2 earners, muliple by 2 
    # HUGE ASSUMPTION
    if income_by_household:
        df['upper_band'] = df['upper_band'] * 2

    df_filt = df[df['upper_band'] > min_wage]  # exclude under certain income value

    # sample from distribution of number of individuals in each wage bracket
    #   iee more chance of sampling from a wage bracket that has higher number of individuals
    sample = df_filt.sample(n=n, weights='n_individuals', replace=True)  
    
    # group/summarise (have n column)
    if group:
        grouped_by_band = sample.groupby('upper_band').size()
        grouped_clean = grouped_by_band.reset_index()
        grouped_clean.rename(columns={0: 'n', 'upper_band': 'income'}, inplace=True)
        wealth_df = grouped_clean
        return wealth_df
    # dont group, have a row for each individual
    else:
        sample_clean = sample[['upper_band']].rename(columns={'upper_band': 'income'}).reset_index(drop=True)
        wealth_df = sample_clean

        # create purchase power as a multiplied value of income (see median multiple)
        wealth_df['purchase_power'] = wealth_df['income'] * affordability_multiplier

        # add risk tolerances
        # uniform distribution
        risk_tolerances, risktol_categories = create_risk_tolerances(n, rt_precision, categorise=True)
        potential_buyers = wealth_df
        potential_buyers['risk_tolerance'] = risk_tolerances
        potential_buyers['risk_tolerance_category'] = risktol_categories  # categorise

        return potential_buyers
    

# update (reduce) price if a home changes in risk category
def price_by_riskcatchange(homes, risk, prev_risk, debug=False,
                     perc_reductions={'None': 0, 'Uninsurable': 10, 'Yellow': 30, 'Red': 100},
                     inplace=False):
    updated_price_homes = copy.deepcopy(homes)  # don't edit original

    home_price_zeroes = np.zeros(len(updated_price_homes))
    
    i_difs = np.where(prev_risk != risk)[0]  # where prev and current risk are not the same
    # create array of both
    #   this will come in handy if I decide to weight changes between groups differently
    #val_difs = np.array([prev_risk[i_difs], cur_risk[i_difs]])  
    #moved_to = val_difs[1]  
    moved_to = risk[i_difs]

    # conditions, when the change meets variables
    conditions = [
        (moved_to == 'None'),
        (moved_to == 'Uninsurable'),
        (moved_to == 'Yellow'),
        (moved_to == 'Red')
    ]

    # create an array of percentage reductions based on passed through dict of reductions by category
    val_reductions = np.select(conditions, perc_reductions.values(), default=0)

    prices_to_change = updated_price_homes['value'][i_difs]  # find unchanged prices at correct indices

    # create new prices by reducing by a percentage
    price_change_values = prices_to_change * (val_reductions / 100)
    #new_prices = prices_to_change - (prices_to_change * (val_reductions / 100))  

    if debug:
        #print(f'{len(i_difs)} houses changed risk category losing a total value of ${int(sum(prices_to_change - new_prices))}')
        if len(i_difs) > 0:
            print(f'{len(i_difs)} houses changed risk category losing a total value of ${int(sum(price_change_values))}')

    #updated_price_homes['value'][i_difs] = new_prices  # append changes at relevent indices

    #return updated_price_homes['value']  # return value only
    home_price_zeroes[i_difs] = price_change_values
    
    return home_price_zeroes


# calculate economic risk of all homes
def calc_econ_risk(homes, precision=2):
    econ_risk = np.around(1 - (homes['value'] / homes['original_value']), precision)

    return econ_risk


# house sale/market system
#   defines if a house is on the market (based on economic risk), then finds
#   potential buyers and decides if the house is sold to any of these buyers
def sales(homes, buyers, max_sale_increase_perc=20, attractiveness_proportion=0.01, 
          attractiveness_moe=0.005, attractiveness_precision=4, buyer_sorting='highest',
          notsold_pricered=0.05, debug=False, return_type='df'):
    
    w_homes = copy.deepcopy(homes)
    w_buyers = copy.deepcopy(buyers)

    all_sale_prices = []
    all_risktol_difs = []

    sold_houses = defaultdict(list)

    # house put on market if the economic risk is greater than what the owner can tolerate
    houses_to_sell = homes[(homes['economic_risk'] > homes['owner_risktolerance']) &
                           (homes['accessible'] == True)]
    
    if len(houses_to_sell) == 0:
        return pd.DataFrame()

    # will convert this to a function later, iterating just while testing
    for i, data in houses_to_sell.iterrows():
        house_sold = False

        price = data['value']
        max_saleprice = price + (price / max_sale_increase_perc)
        # in theory don't need risk category due to this being reflected in economic risk
        econ_risk = data['economic_risk'] 

        # conditions for eligible buyers
        #   1. buyer's purchase power is greater than or equal to the value-based price
        #   2. buyer's purchase power is less than or equal to the maximum sale price
        #   3. buyer's risk tolerance is greater than the economic risk of the house
        eligible_buyers = w_buyers[(w_buyers['purchase_power'] >= price) &
                                 (w_buyers['purchase_power'] <= max_saleprice) &
                                 (w_buyers['risk_tolerance'] > econ_risk)]
        
        # further filter by randomly sampling (interested in buying?)
        if len(eligible_buyers) > 0:
            # filter the eligible buyers by a set proportion
            #   in theory this accounts for geographical location, other constraints that would
            #   prevent viewing a property before going into sale negotiations
            attractiveness_var = np.random.uniform(-attractiveness_moe, attractiveness_moe)
            home_attractiveness = round(attractiveness_proportion + attractiveness_var, attractiveness_precision)
            potential_buyers = eligible_buyers.sample(frac=home_attractiveness)
            
            # randomly determine if a potential buyer is willing to buy based on a set parameter
            """potential_buyers['will_buy'] = np.random.choice([True, False], 
                                            size=len(potential_buyers), 
                                            p=[buyer_purchase_chance, 1 - buyer_purchase_chance])
            """
            # randomly select buyer from available and willing buyers
            if buyer_sorting == 'random':
                #buyers_i = np.where(potential_buyers['will_buy'] == True)[0]  # find buyers that are willing
                
                # if a buyer is available
                """if len(buyers_i > 0):
                    buyer_i = np.random.choice(buyers_i)  # randomly select as willing
                    buyer = potential_buyers.iloc[buyer_i]  # find data of buyer"""
                if len(potential_buyers) > 0:
                    buyer = potential_buyers.sample(n=1, ignore_index=True).iloc[0]
                    
                    # sale price is maximum bid
                    #   equal to the purchase power if that is less than the max price, otherwise
                    #   equal to the max price
                    sale_price = buyer['purchase_power'] if buyer['purchase_power'] < max_saleprice else max_saleprice
                    house_sold = True

            # select highest purchase power buyer from available and willing buyers
            elif buyer_sorting == 'highest':
                
                buyers_bywealth = potential_buyers.sort_values('purchase_power')  # sort buyers by their purchase power
                """buyers_i = np.where(buyers_bywealth['will_buy'] == True)[0]  # find buyers that are willing

                # if a buyer is available
                if len(buyers_i) > 0:
                    buyer = buyers_bywealth.iloc[buyers_i[0]]  # find fist buyer data"""
                if len(buyers_bywealth) > 0:
                    buyer = buyers_bywealth.reset_index().iloc[0]  # temporary
                    # sale price is maximum bid
                    sale_price = buyer['purchase_power'] if buyer['purchase_power'] < max_saleprice else max_saleprice
                    
                    house_sold = True
                
        if house_sold:
            # update main homes array with buyer attributes (replacing seller)
            main_arr_i = data.name

            if return_type == 'df':
                sold_houses['index'].append(main_arr_i)
                sold_houses['value'].append(sale_price)
                sold_houses['purchase_price'].append(sale_price)
                sold_houses['owner_risktolerance'].append(buyer['risk_tolerance'])
                sold_houses['for_sale'].append(False)

                owner_number = w_homes.iloc[main_arr_i]['owner_number']
                owner_number += 1
                sold_houses['owner_number'].append(owner_number)
            
            else:
                w_homes.iloc[main_arr_i]['value'] = sale_price  # try using comma in [] instead of [][]
                w_homes.iloc[main_arr_i]['purchase_price'] = sale_price
                w_homes.iloc[main_arr_i]['owner_risktolerance'] = buyer['risk_tolerance']
                w_homes.iloc[main_arr_i]['owner_number'] += 1  # update occupant number

            # remove buyer from array
            w_buyers = w_buyers.drop(buyer['index'])

            all_sale_prices.append(sale_price)
            all_risktol_difs.append(buyer['risk_tolerance'] - data['owner_risktolerance'])

        else:
            # reduce value
            #print('house not sold')
            main_arr_i = data.name
            lowered_price = (price - (price * notsold_pricered))

            if return_type == 'df':
                # appending nans means the replace will not apply to those columns
                sold_houses['value'].append(lowered_price)
                sold_houses['for_sale'].append(True)
                sold_houses['index'].append(main_arr_i)
                sold_houses['purchase_price'].append(np.nan)
                sold_houses['owner_risktolerance'].append(np.nan)
                sold_houses['owner_number'].append(np.nan)

            else:
                w_homes.iloc[main_arr_i]['value'] = lowered_price

    if debug:
        if len(all_sale_prices) > 0:
            print(f'{len(all_sale_prices)} houses sold, average risk tolerance increase of {round(np.mean(all_risktol_difs), 4)}')
        # total spend of ${sum(all_sale_prices)}

    if return_type == 'df':
        df = pd.DataFrame(sold_houses)
        df.index = df['index']
        return df
    else:
        return w_homes, w_buyers, sold_houses
        

# get all (or filtered) data from neighbours
def get_neighbours_data(main_df, neighbours_indices, all_buildings=True, 
                        filter_data=True, data=['price'], building_index=None):
    if filter_data:
        df_filt = main_df[data]
        df_arr = df_filt.to_numpy()
    else:
        df_arr = main_df.to_numpy()

    if all_buildings:
        all_neighbour_data = [df_arr[n] for n in neighbours_indices]
    else:
        neighbours_indices = neighbours_indices[building_index]
        all_neighbour_data = df_arr[neighbours_indices]
    
    return np.array(all_neighbour_data)


# get average value of neighbours
def get_avg_neighbourval(main_df, neighbours_indices):
    val_arr = main_df['value'].to_numpy()  # speeds things up significantly

    avg_neighbour_val = [np.mean(val_arr[n]) for n in neighbours_indices]

    return avg_neighbour_val


# get differences in home value as a result of the change in neighbourhood values
def modify_neighbour_vals(main_df, prev_df, precision=4):
    curprev_neighbourval_change = (main_df['average_neighbour_value'] - prev_df['average_neighbour_value'])
    val_proportion_increase = (curprev_neighbourval_change / prev_df['average_neighbour_value']) #* 100
    
    val_proportion_increase.fillna(0, inplace=True)
    #print(sum(val_proportion_increase.isna()))
    #print(max(val_proportion_increase), min(val_proportion_increase))
    rnd_changes = np.around(np.random.uniform(0, val_proportion_increase), precision)
    #rnd_changes = np.random.uniform(0, val_proportion_increase)
    return main_df['value'] * rnd_changes


def handle_inaccessibility(main_df, neighbour_arr, inaccess_indices, inaccess_modifier):
    # remove inaccessible homes from neighbour array
    cur_inaccess = len(inaccess_indices)
    new_inaccess_indices = np.array(main_df[main_df['accessible'] == False].index)
    cur_neighbours = [np.setdiff1d(neighbour, inaccess_indices, assume_unique=True) for neighbour in neighbour_arr]

    print(len(new_inaccess_indices) - cur_inaccess, 'newly inaccessible homes')

    all_val_adjustments = []
    for home_i, neighbours in enumerate(cur_neighbours):

        # if any neighbour(s) are gone due to inaccessibility
        if len(neighbours) != len(neighbour_arr[home_i]):
            #self.cur_homes['value'][home_i] *= 1 - np.round(np.random.uniform(0, self.inaccessible_deduction), 3)
            all_val_adjustments.append(1 - np.round(np.random.uniform(0, inaccess_modifier), 3))
        else:
            all_val_adjustments.append(1)  # will multiply current value by 1 to equal original value

    return all_val_adjustments, cur_neighbours, new_inaccess_indices