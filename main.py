# %%
"""
MAIN - instantiate and run the ABM

Notes:
    - Model execution is contained in a loop for sensitivity analysis (i.e. testing many parameter combinations)
    - Requires at least 12GB of RAM, as well as decent single-core performance (not parallel)
    - Be cautious with certain parameters (e.g. slip count, neighbourhood distance) that can greatly increase processing time
    - Has only been tested on Debian server (bullseye). Likely cross-platform issues with file paths

    - Requires an input folder with the following files (see parse_data.py):
        - buildings_council.gpkg  buildings with valuations and geometry
        - cliff_region.gpkg       regions (polygons) where homes are considered in range of cliff erosion
        - cliffs.gpkg             cliff geometry
        - coastline.gpkg          coastline geometry
        - extent.gpkg             processing extent
        - ird_wages_2022.csv      income/wealth data
        - parcels.gpkg            land parcel geometry

BASELINE CONDITIONS
- 0.006+/-0.002m monthly erosion
- 1% storm probability
- 250m neighbourhood distance
"""
import os 
#os.chdir(os.getcwd() + '/model/')
import abm
import utilities as utils

import importlib
importlib.reload(abm)

# VARIABLES TO TEST

# model details
n_steps = 2400
save_output = True
save_path = 'output_runs/'

# housing market/behaviour/policy
riskcat_deductions = {'None': 0, 'Uninsurable': 10, 'Yellow': 30, 'Red': 100}
#neighbourhood_distance = 250
inaccess_deductions = 0.2
risk_cutoffs = [3, 1, 0.5]

# geomorphology/erosion
erosion_dist_var = [0.006, 0.002] #[50, 0]#[0.006, 0.002] #[10, 5]
monthly_storm_probability = 1 # 0, 1, 2, 5, 10, 50
n_slips = 10
n_slips_moe = 5
slip_size = 5
slip_size_moe = 2

# ( already done: 7cm/y), 12cm/y, 20cm/y, 50cm/y, 1m/y, 5m/y 
# (0.01, 0.003), (0.016, 0.004), (0.041, 0.01), (0.083, 0.05), 
#for nd in [0, 100, 250, 500, 1000, 5000]:  
for nd in [5000]:
    # instantiate the model
    cur_model = abm.ABM(
            min_house_area=40,              # minimum area a building must be to be considered a home
            cutoffs=risk_cutoffs,            # distance a risk category is applied [uninsurable, yellow sticker, red sticker]
            neighbourhood_dist=nd,         # distance at which to search nearby houses for price changes
            monthly_inflation_rate=0,        # positive inflation of house prices each month (in %)
            monthly_erosion_dist=erosion_dist_var[0],     # monthly constant erosion rate
            monthly_erosion_moe=erosion_dist_var[1],    # margin of error for erosion rate (will modify the erosion rate with rng)
            erosion_dist_precision=5,       # n decimal places to round erosion to
            cliff_erosion_res=0.001,        # resolution at which to simplify cliffs after erosion
            monthly_storm_probability=monthly_storm_probability,    # chance a storm occurs on any month, percentage
            n_slips=n_slips,                      # number of slips during a storm event
            n_slips_moe=n_slips_moe,                  # +/- margin of error for n_slips
            slip_size=slip_size,                    # size of slips during a storm event
            slip_size_moe=slip_size_moe,                # +/- margin of error for slip_size
            neighbour_history_length=2,     # amount of steps that history of prices/neighbour data is kept
            n_potential_buyers=500000,      # number of potential buyers in market
            afford_multiplier=8,            # amount to multiply median income to consider a home to be within buying range
            use_household_income=True,      # multiply income data by 2 to calculate affordability? (household = 2 income generally)
            riskcat_pricereductions_perc=riskcat_deductions,  # % penalties of house value when moving to risk category
            risktolerance_precision=2,      # decimal places of risk tolerance THIS IS BROKEN 15/09/23
            parsed_data_dir='data/input/',
            inaccess_deduction=inaccess_deductions  # proportion to potentially remove from house value if neighbour becomes inaccessible
    )

    # step the model
    for i, var in enumerate(range(n_steps)):
        with utils.Timer(f'step {i + 1} of {n_steps}'):
            cur_model.step()

        print('\n')

    # save model output
    if save_output:
        out_folder = cur_model.save_model_data(out_dir=save_path)
        print(f'Run saved to {save_path}{out_folder}')

