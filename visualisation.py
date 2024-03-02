# %%
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import pandas as pd
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx
import matplotlib.patches as mpatches
from cycler import cycler
import parse_data as parse
import contextily as cx
import shapely.geometry as shgeo
from numerize.numerize import numerize
from collections import defaultdict
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pickle
from tobler.util import h3fy
from tobler.area_weighted import area_interpolate
import matplotlib as mpl
import matplotlib.colors as mcolors
from matplotlib.patches import RegularPolygon
import math
from shapely.geometry import LineString
from rasterio.plot import show

import visualisation as vis


# setup map
#       add north arrow, scale bar, basemap if requested, etc.
def setup_map(extent_gdf, rows=2, cols=3, figsize=(15, 15), basemap=False, 
              basemap_alpha=0.4, arrow_parameters=(0.05, 0.09, 0.04),
              sharex=False, sharey=False):
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, sharex=sharex, sharey=sharey)

    for i, ax in enumerate(fig.axes):
        scale = ScaleBar(
                dx=1, location='lower right',
                box_alpha=0)
        
        x, y, arrow_length = arrow_parameters #
        ax.annotate('N', xy=(x, y), xytext=(x, arrow_length),
                        arrowprops=dict(facecolor='black', width=3, headwidth=7),
                        ha='center', va='center', fontsize=15,
                        xycoords=ax.transAxes)
        
        extent_gdf.plot(ax=ax, alpha=0)

        if basemap:
                cx.add_basemap(ax, attribution=False, source=cx.providers.OpenStreetMap.Mapnik, 
                                alpha=basemap_alpha, crs='epsg:2193', zoom=16)
        
        ax.add_artist(scale)

        if i > 0 and sharey:
            ax.set(xlabel='Easting')
        else:
            ax.set(xlabel='Easting', ylabel='Northing')

    return fig, axes 


# maps of homes symbolized by risk category
def map_by_riskcat(gdf, figsize=(15, 15), basemap=False, basemap_alpha=0.4):
    #fig, ax = plt.subplots(figsize=figsize)
    fig, ax = setup_map(gdf, figsize=figsize, basemap=basemap, 
                        basemap_alpha=basemap_alpha)
    
    cmap = {'Uninsurable': 'blue', 
            'Yellow': 'orange', 
            'Red': 'red', 
            'None': 'gray'}

    for label, data in gdf.groupby('risk_category'):
            data.plot(ax=ax, color=cmap[label])

            legend_patches = [mpatches.Patch(color=color, label=key) for key, color in cmap.items()]
            ax.legend(handles=legend_patches)

    ax.set(title='Homes by risk category')

    return fig, ax


# get strings of variable values by run id file name
def vars_from_runid(run_id):
    file_split = run_id.split('_')
    erosion_rate = file_split[1].strip('m')
    storm_freq = file_split[2].strip('pc')
    steps = file_split[3].strip('stp')

    return erosion_rate, storm_freq, steps 


"""
plot trends of EITHER home values OR risk tolerance over time, seperated plots grouped by home locations
"""
def groupvar_cliffnoncliff_grouped(data_container, plot_variable, group_variable, legend_titles,
                                   ylabels): 

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), 
                        gridspec_kw={'wspace': 0.1, 'hspace': 0.3},
                        sharey=True)

    for var in ax.flatten():
        var.set_prop_cycle(custom_cycler)

    for i, (home_types, data) in enumerate(data_container.items()):
        for storm_freq, df in data.items():
            df.plot(x='step', y=plot_variable, ax=ax[i], label=storm_freq)
        ax[i].set(xlabel='Month', ylabel=ylabels[plot_variable])

    # set labels, shared legend
    ax[0].set(title='All homes')
    ax[1].set(title='Homes in cliffed areas')
    h, l = ax[0].get_legend_handles_labels()
    for ax in ax: ax.get_legend().remove()
    plt.legend(h,l, loc=(1.05, 0.3), title=legend_titles[group_variable])
    
    return fig, ax  


"""
plot trends of EITHER home values OR risk tolerance over time, seperated plots grouped by home locations
"""
def bothvars_cliffnoncliff_grouped(data_container, plot_variable, legend_titles,
                                   ylabels, group_titles, ab_labels): 

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), 
                        gridspec_kw={'wspace': 0.1, 'hspace': 0.2},
                        sharey=True, sharex=True)

    for var in axes.flatten():
        var.set_prop_cycle(custom_cycler)

    for i, (erosionvar, home_dict) in enumerate(data_container.items()):
        for x, (home_type, data) in enumerate(home_dict.items()):
            for val, df in data.items():
                df.plot(x='step', y=plot_variable, ax=axes[i][x], label=val)

    # set labels, shared legend
    h_gradual, l_gradual = axes[0][0].get_legend_handles_labels()
    h_stormprob, l_stormprob = axes[1][0].get_legend_handles_labels()

    for i, ax in enumerate(axes.flatten()):
        ax.get_legend().remove()
        if i % 2 == 0:
            ax.set(title=group_titles['all_homes'])
        else:
            ax.set(title=group_titles['cliff_homes'])

    #for axes in axes: axes.get_legend().remove()

    fig.legend(h_gradual, l_gradual, bbox_to_anchor=(1.03, 0.8), title=legend_titles['gradual'])
    fig.legend(h_stormprob, l_stormprob, bbox_to_anchor=(1.03, 0.4), title=legend_titles['stormprob'])

    axes[0][0].set(ylabel=ylabels[plot_variable])
    axes[1][0].set(ylabel=ylabels[plot_variable], xlabel='Step')
    axes[1][1].set(xlabel='Step')

    fig.text(0.03, 0.71, ab_labels['a'], fontsize=18, weight='bold' if ab_labels['bold'] else 'normal')
    fig.text(0.03, 0.3, ab_labels['b'], fontsize=18, weight='bold' if ab_labels['bold'] else 'normal')

    return fig, axes  



"""
plot trends of home values over time by variable type, seperated plots by grouped home locations
"""
def groupvar_cliffnoncliff_individual(data_container, group_variable, titles, plot_variable,
                                      ylabels):
    
    fig, ax = plt.subplots(1, 4, figsize=(20, 4), sharey=True)

    for var in ax.flatten():
        var.set_prop_cycle(custom_cycler)

    if group_variable == 'stormprob':
        chosen_vals = ['0', '1', '5', '10']
    elif group_variable == 'gradual':
        chosen_vals = ['0.006', '0.01', '0.016', '0.083']

    for i, var in enumerate(chosen_vals):
        data_container['all_homes'][var].plot(ax=ax[i], x='step', y=plot_variable, label='All homes')
        data_container['cliff_homes'][var].plot(ax=ax[i], x='step', y=plot_variable, label='Homes in cliff area')
        ax[i].set(xlabel='Month', title=var + ' ' + titles[group_variable])

    ax[0].set(ylabel=ylabels[plot_variable])
    #fig.text(0.5, 0.97, f'{stat_type.title()} house value by monthly storm probability', ha='center', fontsize=18)
    h, l = ax[0].get_legend_handles_labels()
    for ax in ax: ax.get_legend().remove()
    plt.legend(h,l, loc=(-1.6, 1.13), title='Home group')

    return fig, ax


def bothvar_cliffnoncliff_individual(data_container, group_titles, plot_variable,
                                      ylabels, group_colours, ab_labels, plot_tippingpoints=False):
    
    fig, ax = plt.subplots(2, 4, figsize=(20, 8), sharey='row', sharex=True,
                           gridspec_kw={'hspace': 0.3, 'wspace': 0.1})

    for var in ax.flatten():
        var.set_prop_cycle(custom_cycler)

    chosen_stormprob_vals = ['0', '1', '5', '10']
    chosen_gradual_vals = ['0.006', '0.01', '0.016', '0.083']

    chosen_vals = [chosen_gradual_vals, chosen_stormprob_vals]

    stormprob_title_suffix = '% storm probability'
    gradual_title_suffix = ' m gradual erosion'

    for i, (erosionvar, home_dict) in enumerate(data_container.items()):
        for x, var in enumerate(chosen_vals[i]):
            data_container[erosionvar]['all_homes'][var].plot(ax=ax[i][x], x='step', y=plot_variable, label=group_titles['all_homes'], 
                                                              color=group_colours['all_homes'])
            data_container[erosionvar]['cliff_homes'][var].plot(ax=ax[i][x], x='step', y=plot_variable, label=group_titles['cliff_homes'],
                                                                color=group_colours['cliff_homes'])
            ax[i][x].set(title=f'{var}{stormprob_title_suffix if erosionvar == "stormprob" else gradual_title_suffix}')

    ax[0][0].set(ylabel=ylabels[plot_variable])
    ax[1][0].set(ylabel=ylabels[plot_variable])

    for i in range(4):
        ax[1][i].set(xlabel='Step')
    #fig.text(0.5, 0.97, f'{stat_type.title()} house value by monthly storm probability', ha='center', fontsize=18)
    
    h, l = ax[0][0].get_legend_handles_labels()

    for i, axis in enumerate(ax.flatten()):
        axis.get_legend().remove()

    # central legend
    if plot_tippingpoints:
        h.append(Line2D([0], [0], color='dimgrey', label='Tipping point', linestyle='--'))
        l.append('Tipping point')    

    plt.legend(h,l, loc=(-1.4, -0.5))
    
    fig.text(0.07, 0.71, ab_labels['a'], fontsize=18, weight='bold' if ab_labels['bold'] else 'normal')
    fig.text(0.07, 0.3, ab_labels['b'], fontsize=18, weight='bold' if ab_labels['bold'] else 'normal')

    if plot_tippingpoints:
        if plot_variable == 'median_value':
            val_tippoints = [1850, 1300, 900, 160, None, 1900, 1550, 1180]
            for i, axis in enumerate(ax.flatten()):
                if val_tippoints[i] != None:
                    axis.axvline(val_tippoints[i], linestyle='--', alpha=0.7, color='dimgrey')
        else:
            rt_tippoints = [800, 550, 400, 100, 800, 750, 800, 400]
            for i, axis in enumerate(ax.flatten()):
                if rt_tippoints[i] != None:
                    axis.axvline(rt_tippoints[i], linestyle='--', alpha=0.7, color='dimgrey')

    return fig, ax


"""
plot trends of house values over time, including lines for each variable. plots seperated by mean, median, and grouped home locations
"""
def all_values_grouped(data_container, group_variable, legend_titles, titles):
    fig, ax = plt.subplots(2, 2, figsize=(16, 14), gridspec_kw={'wspace': 0.2, 'hspace': 0.3})

    for var in ax.flatten():
        var.legend(title=legend_titles[group_variable])
        var.set_prop_cycle(custom_cycler)

    for storm_freq, df in data_container['all_homes'].items():
        df.plot(x='step', y='median_value', ax=ax[0][0], label=storm_freq)
        df.plot(x='step', y='mean_value', ax=ax[0][1], label=storm_freq)

    for storm_freq, df in data_container['cliff_homes'].items():
        df.plot(x='step', y='median_value', ax=ax[1][0], label=storm_freq)
        df.plot(x='step', y='mean_value', ax=ax[1][1], label=storm_freq)

    ax[0][0].set(xlabel='Step', ylabel='Median value ($)', title=f'Median value over time by {titles[group_variable]}')
    ax[0][1].set(xlabel='Step', ylabel='Mean value ($)', title=f'Mean value over time by {titles[group_variable]}')
    ax[1][0].set(xlabel='Step', ylabel='Median value ($)', title=f'Median value over time by {titles[group_variable]}')
    ax[1][1].set(xlabel='Step', ylabel='Mean value ($)', title=f'Mean value over time by {titles[group_variable]}')

    fig.text(0.5, 0.91, 'All homes in Devonport', fontsize=20, ha='center')
    fig.text(0.5, 0.48, 'Homes in cliff area', fontsize=20, ha='center')
    fig.text(0.5, 0.07, 'Gradual erosion rate: 0.006m +/- 0.002m', ha='center', fontsize=14)

    return fig, ax


"""
plot change in variable from start to end of model, aggregated by sa1 
"""
def spatial_comparison(run_id, extent_path, cliff_region_path, title=True):
    gdf = gpd.read_file(f'output_runs/{run_id}/home_changes_sa1.gpkg')

    #gdf.plot(column='value')
    fig, axes = setup_map(gpd.read_file(extent_path), basemap=False, figsize=(15, 10),
                            rows=1, cols=2)

    gdf.plot(ax=axes[0], column='value', 
            legend_kwds={'shrink': 0.5, 'label': 'Average change in value from year 1 to 200 (%)', 
                        'orientation': 'horizontal', 'location': 'bottom', 'pad': 0.08}, 
            legend=True, cmap='cividis', vmax=0, vmin=-100,
            edgecolor='gray', linewidth=0.3, zorder=0, alpha=0.8)
    axes[0].set(title='Home value')

    gdf.plot(ax=axes[1], column='owner_risktolerance', 
            legend_kwds={'shrink': 0.5, 'label': 'Average change in risk tolerance from year 1 to 200 (%)', 
                        'orientation': 'horizontal', 'location': 'bottom', 'pad': 0.08}, 
            legend=True, cmap='viridis_r', vmin=0, vmax=60,
            edgecolor='gray', linewidth=0.3, zorder=0, alpha=0.8)
    axes[1].set(title='Homeowner risk tolerance')

    for ax in axes.flatten():
        cliff_regions = gpd.read_file(cliff_region_path)
        cliff_regions.plot(ax=ax, facecolor='none', edgecolor='black', zorder=1, linestyle='--',
                        linewidth=1.5, alpha=0.5)

        avg_leg = [Patch(facecolor='none', edgecolor='black', linestyle='--', label='Cliff erosion zones')]
        ax.legend(handles=avg_leg, loc='upper right')

    if title:
        erosion_rate, storm_freq, steps = vars_from_runid(run_id)
        fig.text(0.5, 0.95, f'Change in key variables from starting conditions and {int(int(steps) / 12)} years in future\n' \
                f'Monthly erosion of {erosion_rate}m, storm probability of {storm_freq}%',
                ha='center', fontsize=16)
    
    return fig, ax


"""
plot change in cliff geometry change between 2 step values
"""
def geography_changes_stanleypt(runid_1, runid_2, plot_area='stanleypt', geom_path='data/input/buildings_council.gpkg',
                                basemap_alpha=0.6, n_arrow_params=(0.05, 0.09, 0.04)):

    geog_1 = parse.read_land_geography(runid_1)[-1:]
    geog_2 = parse.read_land_geography(runid_2)[-1:]

    home_geom = gpd.read_file(geom_path)

    if plot_area == 'bayswater':
        bbox = [1757327.9279 , 5923395.3466, 1758961.8308, 5924566.1155]
        bayswater_poly = shgeo.Polygon([bbox[2:], (bbox[:2][0], bbox[2:][1]), bbox[:2], (bbox[2:][0], bbox[:2][1])])
        area = bayswater_poly
    elif plot_area == 'stanleypt':
        bbox = [1758022.7954,5922562.8052,1758746.1331,5923163.1678]
        stanleypt_poly = shgeo.Polygon([bbox[2:], (bbox[:2][0], bbox[2:][1]), bbox[:2], (bbox[2:][0], bbox[:2][1])])
        area = stanleypt_poly
    elif area == 'eastern':
        bbox = [1759445.2755,5924614.2879,1760588.7690,5926156.4222]
        eastern_poly = shgeo.Polygon([bbox[2:], (bbox[:2][0], bbox[2:][1]), bbox[:2], (bbox[2:][0], bbox[:2][1])])
        area = eastern_poly

    #fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    fig, ax = setup_map(gpd.GeoSeries(area), rows=1, cols=2, figsize=(17, 7), basemap=True, 
                        basemap_alpha=basemap_alpha, arrow_parameters=n_arrow_params,
                        sharey=True)

    geog_1.clip(area).plot(facecolor='none', edgecolor='none', ax=ax[1])  # set extent
    geog_2.clip(area).plot(ax=ax[1], facecolor='none', edgecolor='black', linewidth=1)

    geog_2.set_geometry('cliffs').clip(area).plot(color='red', ax=ax[1], linewidth=1.5)
    home_geom.clip(area).plot(ax=ax[1], facecolor='gray', edgecolor='black')

    geog_1.clip(area).plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=1)
    geog_1.set_geometry('cliffs').clip(area).plot(color='red', ax=ax[0], linewidth=1.5)
    home_geom.clip(area).plot(ax=ax[0], facecolor='gray', edgecolor='black')

    runid1_erosionrate, runid1_stormfreq, _ = vars_from_runid(runid_1)
    ax[0].set(title=f'{runid1_erosionrate} m monthly erosion,\n{runid1_stormfreq}% monthly storm probability')

    runid2_erosionrate, runid2_stormfreq, _ = vars_from_runid(runid_2)
    ax[1].set(title=f'{runid2_erosionrate} m monthly erosion,\n{runid2_stormfreq}% monthly storm probability')

    for axis in ax.flatten():
        cliff_leg = [Line2D([0], [0], color='red', label='Cliff edges'),
                     Patch(facecolor='gray', edgecolor='black', linestyle='-', label='Houses')]
        axis.legend(handles=cliff_leg, loc='upper left')
    
    return fig, ax 



def geography_changes_stanleypt_v2(runid_1, runid_2, plot_area='stanleypt', geom_path='data/input/buildings_council.gpkg',
                                basemap_alpha=0.6, n_arrow_params=(0.05, 0.09, 0.04)):

    
    runid1_geog2 = parse.read_land_geography(runid_1)[-1:]
    runid1_geog1 = parse.read_land_geography(runid_1)[:1]
    runid2_geog2 = parse.read_land_geography(runid_2)[-1:]

    home_geom = gpd.read_file(geom_path)

    if plot_area == 'bayswater':
        bbox = [1757327.9279 , 5923395.3466, 1758961.8308, 5924566.1155]
        bayswater_poly = shgeo.Polygon([bbox[2:], (bbox[:2][0], bbox[2:][1]), bbox[:2], (bbox[2:][0], bbox[:2][1])])
        area = bayswater_poly
    elif plot_area == 'stanleypt':
        bbox = [1758022.7954,5922562.8052,1758746.1331,5923163.1678]
        stanleypt_poly = shgeo.Polygon([bbox[2:], (bbox[:2][0], bbox[2:][1]), bbox[:2], (bbox[2:][0], bbox[:2][1])])
        area = stanleypt_poly
    elif plot_area == 'eastern':
        bbox = [1759445.2755,5924614.2879,1760588.7690,5926156.4222]
        eastern_poly = shgeo.Polygon([bbox[2:], (bbox[:2][0], bbox[2:][1]), bbox[:2], (bbox[2:][0], bbox[:2][1])])
        area = eastern_poly

    #fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    fig, ax = setup_map(gpd.GeoSeries(area), rows=1, cols=2, figsize=(17, 7), basemap=True, 
                        basemap_alpha=basemap_alpha, arrow_parameters=n_arrow_params,
                        sharey=True)

    for axis in ax.flatten():
        runid1_geog2.clip(area).plot(facecolor='none', edgecolor='none', ax=axis)  # set extent
        home_geom.clip(area).plot(ax=axis, facecolor='gray')
        runid1_geog1.clip(area).plot(ax=axis, facecolor='none', edgecolor='black', linewidth=1)
        runid1_geog1.set_geometry('cliffs').clip(area).plot(color='cornflowerblue', ax=axis, linewidth=2)

    # plot first run
    #runid1_geog1.clip(area).plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=1)
    runid1_geog2.set_geometry('cliffs').clip(area).plot(color='red', ax=ax[0], linewidth=2)
    #home_geom.clip(area).plot(ax=ax[0], facecolor='gray', edgecolor='black')

    # plot second run
    #runid1_geog1.clip(area).plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=1)
    runid2_geog2.set_geometry('cliffs').clip(area).plot(color='red', ax=ax[1], linewidth=2)

    runid1_erosionrate, runid1_stormfreq, _ = vars_from_runid(runid_1)
    ax[0].set(title=f'{runid1_erosionrate} m monthly erosion,\n{runid1_stormfreq}% monthly storm probability')

    runid2_erosionrate, runid2_stormfreq, _ = vars_from_runid(runid_2)
    ax[1].set(title=f'{runid2_erosionrate} m monthly erosion,\n{runid2_stormfreq}% monthly storm probability')

    
    cliff_leg = [Line2D([0], [0], color='cornflowerblue', label='Cliff edge after 1 year'),
                 Line2D([0], [0], color='red', label='Cliff edge after 200 years'),
                 Line2D([0], [0], color='black', label='Cliff erosion zone'),
                    Patch(facecolor='gray', linestyle='-', label='Houses')]
    fig.legend(handles=cliff_leg, loc=(0.45, -0.02), 
               bbox_to_anchor=(0.45, -0.06))
    
    return fig, ax 


"""
plot of total value (sum of property values) and % change between start and end
"""
def total_values(data_container, group_variable, legend_titles):
    totals_container = defaultdict(list)
    for storm_freq, df in data_container['all_homes'].items():
        start_val = df[:1]['total_value'].iloc[0]
        end_val = df[-1:]['total_value'].iloc[0]
        percent_change = ((start_val - end_val) / start_val) * 100
        total_change = numerize(start_val - end_val)

        totals_container['ending_value'].append(end_val)
        totals_container['variable'].append(storm_freq)
        totals_container['percentage_change'].append(percent_change)
        totals_container['total_change'].append(total_change)

    totals_container_cliff = defaultdict(list)
    for storm_freq, df in data_container['cliff_homes'].items():
        start_val = df[:1]['total_value'].iloc[0]
        end_val = df[-1:]['total_value'].iloc[0]
        percent_change = ((start_val - end_val) / start_val) * 100
        total_change = numerize(start_val - end_val)

        totals_container_cliff['ending_value'].append(end_val)
        totals_container_cliff['variable'].append(storm_freq)
        totals_container_cliff['percentage_change'].append(percent_change)
        totals_container_cliff['total_change'].append(total_change)

    df = pd.DataFrame(totals_container)
    df_cliff = pd.DataFrame(totals_container_cliff)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    sns.barplot(df, x='variable', y='percentage_change', ax=ax[0], color='cornflowerblue')
    sns.barplot(df_cliff, x='variable', y='percentage_change', ax=ax[1], color='orange')

    ax[0].set(title='All homes')
    ax[1].set(title='Homes in cliffed areas')

    for var in ax.flatten():
        var.set(xlabel=legend_titles[group_variable].replace('\n', ' '), 
                ylabel='Reduction in overall market value (%)')

    x_offset = -0.45
    y_offset = 0.3
    for i, p in enumerate(ax[0].patches):
        b = p.get_bbox()
        val = '    - ' + totals_container['total_change'][i]
        ax[0].annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), fontsize=9)
    for i, p in enumerate(ax[1].patches):
        b = p.get_bbox()
        val = '    - ' + totals_container_cliff['total_change'][i]
        ax[1].annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), fontsize=9)

    return fig, ax


"""
plot of total value (sum of property values) and % change between start and end
"""
def total_values_both(all_data_container, legend_titles, xlabels, home_titles,
                      colours, ab_labels):

    totals_container_both = defaultdict(dict)
    for var, data_container in all_data_container.items():
        totals_container = defaultdict(list)
        for storm_freq, df in data_container['all_homes'].items():
            start_val = df[:1]['total_value'].iloc[0]
            end_val = df[-1:]['total_value'].iloc[0]
            percent_change = ((start_val - end_val) / start_val) * 100
            total_change = numerize(start_val - end_val)

            totals_container['ending_value'].append(end_val)
            totals_container['variable'].append(storm_freq)
            totals_container['percentage_change'].append(percent_change)
            totals_container['total_change'].append(total_change)

        totals_container_cliff = defaultdict(list)
        for storm_freq, df in data_container['cliff_homes'].items():
            start_val = df[:1]['total_value'].iloc[0]
            end_val = df[-1:]['total_value'].iloc[0]
            percent_change = ((start_val - end_val) / start_val) * 100
            total_change = numerize(start_val - end_val)

            totals_container_cliff['ending_value'].append(end_val)
            totals_container_cliff['variable'].append(storm_freq)
            totals_container_cliff['percentage_change'].append(percent_change)
            totals_container_cliff['total_change'].append(total_change)

        totals_container_both[var]['all'] = totals_container
        totals_container_both[var]['cliff'] = totals_container_cliff

    fig, ax = plt.subplots(2, 2, figsize=(13, 10), sharey='row',
                           gridspec_kw={'hspace': 0.3, 'wspace': 0.1})

    for i, (var, data) in enumerate(totals_container_both.items()):
        
        df = pd.DataFrame(data['all'])
        df_cliff = pd.DataFrame(data['cliff'])


        sns.barplot(df, x='variable', y='percentage_change', ax=ax[i][0], color=colours['all_homes'])
        sns.barplot(df_cliff, x='variable', y='percentage_change', ax=ax[i][1], color=colours['cliff_homes'])

        """ax[0].set(title='All homes')
        ax[1].set(title='Homes in cliffed areas')"""

        """for var in ax.flatten():
            var.set(xlabel=legend_titles[var].replace('\n', ' '), 
                    ylabel='Reduction in overall market value (%)')"""

        if var == 'gradual':
            x_offset = -0.4
            y_offset = 1
        else:
            x_offset = -0.5
            y_offset = 1

        for x, p in enumerate(ax[i][0].patches):
            b = p.get_bbox()
            val = '  -$' + data['all']['total_change'][x]
            ax[i][0].annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), fontsize=9)
        for x, p in enumerate(ax[i][1].patches):
            b = p.get_bbox()
            val = '  -$' + data['cliff']['total_change'][x]
            ax[i][1].annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset), fontsize=9)

    for i, axis in enumerate(ax.flatten()):
        if i % 2 == 0:
            axis.set(ylabel='Proportional reduction in\noverall market value (%)',
                     title=home_titles['all_homes'])
        else:
            axis.set(ylabel='', title=home_titles['cliff_homes'])

        if i in [0, 1]:
            axis.set(xlabel=xlabels['gradual'])
            axis.set_ylim((0, 90))
        else:
            axis.set(xlabel=xlabels['stormprob'])
            axis.set_ylim((0, 80))

    fig.text(0.03, 0.71, ab_labels['a'], fontsize=18, weight='bold' if ab_labels['bold'] else 'normal')
    fig.text(0.03, 0.3, ab_labels['b'], fontsize=18, weight='bold' if ab_labels['bold'] else 'normal')

    return fig, ax


def custom_savefig(fig, name, dpi=200):
    fig.savefig(name, dpi=dpi, facecolor='white', bbox_inches='tight')


# needs to be converted to function
def transect_map(show_output=True):

    coastline_gdf = gpd.read_file('data/input/coastline.gpkg')
    transect_length = 150

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
    import rasterio

    dem_path = 'data/source/dev_dem.tif'

    dem = rasterio.open(dem_path, crs=2193)
    band_object = dem.read(1)

    #fig, ax = plt.subplots(1, figsize=(8, 8))
    if not show_output:
        fig, ax = vis.setup_map(coastline_gdf, rows=1, cols=1, figsize=(8, 8))

        transects_gdf.plot(ax=ax, linewidth=1)
        coastline_gdf.plot(ax=ax, color='black', alpha=0.3)

        show(dem, transform=dem.transform, ax=ax, cmap='Reds', alpha=0.7,
            vmin=0, vmax=75)
        xmin, ymin, xmax, ymax = transects_gdf.total_bounds
        pad = 200  # add a padding around the geometry
        ax.set_xlim(xmin-pad, xmax+pad)
        ax.set_ylim(ymin-pad, ymax+pad)

        img = ax.imshow(band_object, 
                                cmap='Reds', 
                                vmin=0, 
                                vmax=75)
        fig.colorbar(img, ax=ax, shrink=0.7, label='Elevation (m)')

        cliff_leg = [Line2D([0], [0], color='cornflowerblue', label='Transects'),
                    Line2D([0], [0], color='black', alpha=0.5, label='Coastline')]
        ax.legend(handles=cliff_leg, loc='upper right')
    
    else:
        fig, ax = vis.setup_map(coastline_gdf, rows=1, cols=2, figsize=(20, 8), 
                                sharey=True, sharex=True)

        # transects
        transects_gdf.plot(ax=ax[0], linewidth=1)
        coastline_gdf.plot(ax=ax[0], color='black', alpha=0.3)

        show(dem, transform=dem.transform, ax=ax[0], cmap='Reds', alpha=0.7,
            vmin=0, vmax=75)
        xmin, ymin, xmax, ymax = transects_gdf.total_bounds
        pad = 200  # add a padding around the geometry
        ax[0].set_xlim(xmin-pad, xmax+pad)
        ax[0].set_ylim(ymin-pad, ymax+pad)

        img = ax[0].imshow(band_object, 
                                cmap='Reds', 
                                vmin=0, 
                                vmax=75)
        fig.colorbar(img, ax=ax[0], shrink=0.7, label='Elevation (m)')

        cliff_leg = [Line2D([0], [0], color='cornflowerblue', label='Transects'),
                    Line2D([0], [0], color='black', alpha=0.5, label='Coastline')]
        ax[0].legend(handles=cliff_leg, loc='upper right')

        # output
        cliffs = gpd.read_file('data/input/cliffs.gpkg')
        coastline_gdf.plot(ax=ax[1], color='black', linewidth=0.5)
        cliffs.plot(ax=ax[1], color='#d5b43c', linewidth=5)
        cliff_leg = [Line2D([0], [0], color='#d5b43c', label='Cliff edges'),
                    Line2D([0], [0], color='black', alpha=0.5, label='Coastline')]
        ax[1].legend(handles=cliff_leg, loc='upper right')
        
        ax[0].set_title('A',fontweight="bold", size=18)
        ax[1].set_title('B',fontweight="bold", size=18)

        plt.subplots_adjust(wspace=-0.1)

    return fig, ax


# plot of final geography
def final_landmasses(runid_1, runid_2, figsize=(16, 8)):
    runs = [runid_1, runid_2]
    
    study_area = gpd.read_file('data/input/extent.gpkg')

    fig, ax = setup_map(study_area, rows=1, cols=2, figsize=(16, 8))

    for i, run in enumerate(runs):
        run_geog = pd.read_pickle(f'output_runs/{run}/geography.pkl')[-1:]
        run_final_geog = gpd.GeoDataFrame(run_geog, geometry='land_geom')
        study_area.plot(facecolor='none', edgecolor='dimgrey',
                        linewidth=1.5, ax=ax[i], linestyle='--')
        run_final_geog.plot(facecolor='none', edgecolor='lightcoral',
                        linewidth=1.5, ax=ax[i])

        cliff_leg = [Line2D([0], [0], color='dimgrey', label='Original land area'),
                    Line2D([0], [0], color='lightcoral', label='Land area after 200 years'),
                    Patch(facecolor='gray', edgecolor='black', linestyle='-', label='Houses')]
        ax[i].legend(handles=cliff_leg, loc='upper right')

    return fig, ax 


# kde plot/histogram of risk tolerance for homes with zero value (inaccessible)
# can output histogram of final changes or kde plot (uses same data parsing loop)
def zeroval_rt_kde(plot_type, xlabels, legend_titles, titles):
    """
    #if no pickle file, run the following: 
    
    container = defaultdict(dict)

    for var in ['gradual', 'stormprob']:
        if var == 'gradual':
            data = gradualrate_runs
        else:
            data = stormprob_runs

        for i, x in enumerate(data):
            df = pd.read_pickle('output_runs/' + x + '/homes.pkl')
            #df_filt = df.loc[df['value'] == 0]
            df_filt = df.loc[df['in_cliff_area'] == False]
            del df

            avg_risktol = df_filt['owner_risktolerance'].mean()

            rate, stormprob, _ = vars_from_runid(x)

            #container[var][rate if var == 'gradual' else stormprob] = df_filt
            container[var][rate if var == 'gradual' else stormprob] = avg_risktol
            del df_filt
    
    """
    zeroval_open_read = open("output_runs/misc_aggregations/zeroval_dfs.pkl", "rb")
    c = pickle.load(zeroval_open_read)

    if plot_type == 'kde':
        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))

        for var in ax.flatten():
            var.set_prop_cycle(custom_cycler)

        #sns.histplot(data=planets, x="distance", log_scale=True, element="step", fill=False)

        for i, var in enumerate(['gradual', 'stormprob']):
            for rate, df in c[var].items():
                sns.kdeplot(data=df, x='owner_risktolerance', label=rate, ax=ax[i])
                #sns.histplot(data=df, x='owner_risktolerance', label=rate, ax=ax, fill=False, bins=bins)
            ax[i].set(xlabel=xlabels['mean_rt'])

        ax[0].legend(title=legend_titles['gradual'])
        ax[0].set(title=titles['gradual'])

        ax[1].legend(title=legend_titles['gradual'])
        ax[1].set(title=titles['stormprob'])

        return fig, ax

    else:
        container_agg = defaultdict(dict)
        for x in ['gradual', 'stormprob']:
            container_agg[x] = defaultdict(list)
            for rate, df in c[x].items():
                agg = np.mean(df['owner_risktolerance'])
                container_agg[x]['rate'].append(rate)
                container_agg[x]['mean'].append(agg)

        fig, ax = plt.subplots(ncols=2, figsize=(16, 6))

        df = pd.DataFrame(container_agg['gradual'])
        sns.barplot(df, x='rate', y='mean', ax=ax[0], color='cornflowerblue')

        df = pd.DataFrame(container_agg['stormprob'])
        sns.barplot(df, x='rate', y='mean', ax=ax[1], color='orange')

        ax[0].set(ylim=(0.5, 1), title='Gradual erosion', xlabel='Gradual erosion rate (m)',
                ylabel='Mean (inaccessible) homeowner risk tolerance')

        ax[1].set(ylim=(0.5, 1), title='Storm erosion', xlabel='Storm probability (%)',
                ylabel='Mean (inaccessible) homeowner risk tolerance')

        return fig, ax


"""
plot total number of house sales by both erosion types and all rates
"""
def total_sales(group_colours, home_titles, xlabels, titles):

    fig, ax = plt.subplots(ncols=2, figsize=(12, 6), sharey=False)

    for i, axis in enumerate(ax.flatten()):
        if i == 0:
            data = parse.load_all_summarised_runs(gradualrate_runs, group_by='gradual')
        else:
            data = parse.load_all_summarised_runs(stormprob_runs, group_by='stormprob')

        c = defaultdict(list)
        for home_type in ['all_homes', 'cliff_homes']:
            for var_value in data[home_type].keys():
                df = data[home_type][var_value]
                final_mean_ownernum = max(df['mean_ownernum'])
                final_total_ownernum = max(df['total_ownernum'])

                c['variable'].append(var_value)
                c['total_ownernum'].append(final_total_ownernum)
                c['mean_ownernum'].append(final_mean_ownernum)
                c['home_type'].append(home_type)

        df = pd.DataFrame(c)

        sns.barplot(df.loc[df['home_type'] == 'all_homes'], x='variable', 
                    y='total_ownernum', color=group_colours['all_homes'], ax=axis,
                    label=home_titles['all_homes'])

        sns.barplot(df.loc[df['home_type'] == 'cliff_homes'], x='variable', 
                    y='total_ownernum', color=group_colours['cliff_homes'], ax=axis, 
                    label=home_titles['cliff_homes'])

        if i == 0:
            axis.set(xlabel=xlabels['gradual'], ylabel='Total number of home sales', title=titles['gradual'])
        else:
            axis.set(xlabel=xlabels['stormprob'], ylabel='', title=titles['stormprob'])

    plt.legend(loc=(-0.35, -0.25))

    return fig, ax


"""
create a sample distance matrix to put in table to describe process
"""
def create_sample_distance_matrix():
    from scipy.spatial import distance_matrix

    def get_neighbours(x_geom):
        geoms_arr = np.array([(point.x, point.y) for point in x_geom])

        dists = distance_matrix(geoms_arr, geoms_arr)

        return dists

    homes = gpd.read_file('data/homes_geom_filt40.gpkg', crs=2193)
    homes = homes[:5]

    matrix = get_neighbours(list(homes.geometry.centroid))

    return matrix


def income_wealth_example():
    # get wages/income = wealth
    df = pd.read_csv('data/source/ird_wages_2022.csv')

    # filter upper band data (for IRD spreadsheet)
    def filter_upperband(x, max_limit=300000):
            i = x.replace('$', '').replace(',', '').replace('Over ', '').replace(' ', '').split('-')
            return i[1] if len(i) > 1 else 300000

    df['upper_band'] = pd.to_numeric(df['band'].apply(filter_upperband))  # convert to numeric



    df['upper_band'] = df['upper_band'] * 2

    df_filt = df[df['upper_band'] > 10000]  # exclude under certain income value

    df = df_filt
    # Creating a new DataFrame with replicated rows based on 'n_individuals'
    df_with_n_columns = df.loc[df.index.repeat(df['n_individuals'])].reset_index(drop=True)

    # Dropping the 'n_individuals' column as it's no longer needed
    df_with_n_columns = df_with_n_columns.drop(columns='n_individuals')

    import numpy as np
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    df_with_n_columns.plot(y='upper_band', kind='hist', ax=ax[0], legend=False, bins=np.arange(0, 600000, 25000))
    ax[0].set(xlabel='Household income ($)', ylabel='Number of households', title='Filtered and multiplied income data')

    for bar in ax[0].patches:
        bar.set_edgecolor('black')

    sample = df_filt.sample(n=10000, weights='n_individuals', replace=True)  

    sample_clean = sample[['upper_band']].rename(columns={'upper_band': 'income'}).reset_index(drop=True)
    wealth_df = sample_clean

    wealth_df.plot(kind='hist', ax=ax[1], legend=False, bins=np.arange(0, 600000, 25000), color='orange')
    ax[1].set(xlabel='Household income ($)', ylabel='Number of households', title='Sample from filtered income data')

    for bar in ax[1].patches:
        bar.set_edgecolor('black')

    return fig, ax 


def poststormimpact(container, xlabels, ylabels, titles):
    df = container['all_homes']['0.006']

    df_storms = df.loc[df['storm'] == True]

    storm_steps = list(df_storms['step'])

    # 50 years = 600 steps
    plot_range = (1800, 2400)

    fig, ax = plt.subplots(ncols=2, figsize=(14, 5), gridspec_kw={'wspace': 0.2})

    df_inrange = df.loc[(df['step'] >= plot_range[0]) & (df['step'] < plot_range[1])]

    df_inrange.plot(x='step', y='n_inaccessible', ax=ax[0], legend=False)
    ax[0].set(ylabel='Number of inaccessible homes', title='Home inaccessibility')

    df_inrange.plot(x='step', y='mean_rt', ax=ax[1], legend=False)
    ax[1].set(ylabel=ylabels['mean_rt'], title=titles['mean_rt'])

    for axis in ax.flatten():
        for x in storm_steps:
            if plot_range[0] <= x <= plot_range[1]:
                axis.axvline(x, linestyle='--', alpha=0.7, color='gray')

    cliff_leg = [Line2D([0], [0], color='gray', label='Storm event during step', alpha=0.7, linestyle='--')]
    fig.legend(handles=cliff_leg, loc=(0.4, -0.02),
                bbox_to_anchor=(0.42, -0.06))

    # annotation A
    lw = 1.5
    fontsize = 15
    colour = 'brown'
    ax[0].annotate('', xy=(1900, 80), xytext=(1800, 80), 
        xycoords='data', textcoords='data',
        arrowprops={'arrowstyle': '|-|', 'linewidth': lw, 'color': colour})
    ax[0].annotate('A', xy=(1850, 82), ha='center', va='center', fontsize=fontsize,
                color=colour)

    ax[1].annotate('', xy=(1900, 0.53), xytext=(1800, 0.53), 
        xycoords='data', textcoords='data',
        arrowprops={'arrowstyle': '|-|', 'linewidth': lw, 'color': colour})
    ax[1].annotate('A', xy=(1850, 0.5307), ha='center', va='center', fontsize=fontsize,
                color=colour)

    # annotation B
    ax[0].annotate('', xy=(2150, 80), xytext=(2300, 80), 
        xycoords='data', textcoords='data',
        arrowprops={'arrowstyle': '|-|', 'linewidth': lw, 'color': colour})
    ax[0].annotate('B', xy=(2225, 82), ha='center', va='center', fontsize=fontsize,
                color=colour)

    ax[1].annotate('', xy=(2150, 0.53), xytext=(2300, 0.53), 
        xycoords='data', textcoords='data',
        arrowprops={'arrowstyle': '|-|', 'linewidth': lw, 'color': colour})
    ax[1].annotate('B', xy=(2225, 0.5307), ha='center', va='center', fontsize=fontsize,
                color=colour)
    
    ax[0].set(xlabel=xlabels['step'])
    ax[1].set(xlabel=xlabels['step'])
    
    return fig, ax 


def stats_by_spatialunit_hex(homes_archive_df, step_a, step_b, 
                         geom_path, hex,
                         vars=['value', 'owner_risktolerance', 'average_neighbour_value']):

    #all_homes = copy.deepcopy(homes_archive_df)
    all_homes = homes_archive_df

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
    homes_areas = gpd.sjoin(homes_diff_gdf, hex, how='left', predicate='intersects')
    homes_areas = homes_areas[vars + ['index_right']]
    sa1_home_diff = homes_areas.groupby('index_right').mean().reset_index()
    sa1_home_diff['hex_id'] = sa1_home_diff['index_right']

    # create gdf

    sa1_home_gdf = gpd.GeoDataFrame(sa1_home_diff.merge(hex, right_index=True,
                                                        left_on='index_right'))

    # fill na values with -100 for avg neighbour values (when no more neighbours)
    sa1_home_gdf['average_neighbour_value'] = sa1_home_gdf['average_neighbour_value'].fillna(-100)

    return sa1_home_gdf, hex


def create_hexed_map(area, gdf_gradual, gdf_storm, hex, var):

    if var == 'value':
        vmin, vmax = -100, 0
        cmap = 'cividis'
        legend_title = 'Average change in home value over model run (%)\n(gray=no homes/data)'
    else:
        vmin, vmax = 0, 60
        cmap = 'cividis_r'
        legend_title = 'Average change in homeowner risk tolerance over model run (%)\n(gray=no homes/data)'
        

    fig, ax = setup_map(area, rows=1, cols=2, figsize=(15, 8))

    background_facecolor = 'none'

    gdf_gradual.plot(column=var, vmax=vmax, vmin=vmin, cmap=cmap, legend=False, 
            ax=ax[0], edgecolor='gray', linewidth=0.2)
    hex.plot(ax=ax[0], facecolor=background_facecolor, edgecolor='gray', linewidth=0.2, 
            zorder=0, alpha=0.8)
    ax[0].set_title('A',fontweight="bold", size=18)

    gdf_storm.plot(column=var, vmax=vmax, vmin=vmin, cmap=cmap, legend=False, 
            ax=ax[1], edgecolor='gray', linewidth=0.2)
    hex.plot(ax=ax[1], facecolor=background_facecolor, edgecolor='gray', linewidth=0.2, 
            zorder=0, alpha=0.8)
    ax[1].set_title('B',fontweight="bold", size=18)

    # Normalize the color scale to range from 0 to 100

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Create a ScalarMappable object
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Set an empty array to use the Normalize object

    # Create a colorbar
    #cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cax = fig.add_axes([0.35, 0.03, 0.31, 0.03])
    cbar = plt.colorbar(sm, cax=cax, orientation='horizontal',
                        label=legend_title)

    for axis in ax.flatten():
        cliff_regions = gpd.read_file('data/input/cliff_region.gpkg')
        cliff_regions.plot(ax=axis, facecolor='none', edgecolor='black', zorder=1, linestyle='--',
                        linewidth=1.5, alpha=0.3)

    avg_leg = [Patch(facecolor='none', edgecolor='black', linestyle='--', label='Cliff erosion zones')]
    fig.legend(handles=avg_leg, bbox_to_anchor=(0.3, 0.05))

    return fig, ax


if __name__ == '__main__':

    """
    SET UP PLOTTING AND VIS CONSTANTS
    """
    # colours
    # taken from https://gist.github.com/thriveth/8560036
    colourblind_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                    '#f781bf', '#a65628', '#984ea3',
                    '#999999', '#e41a1c', '#dede00']
    custom_cycler = (cycler(color=colourblind_cycle))

    homegroup_colours = {'all_homes': 'plum',
            'cliff_homes': 'mediumseagreen'}

    # titles and labels 
    legend_titles = {'gradual': 'Gradual erosion\nrate (m)',
            'stormprob': 'Storm\nprobability (%)'}
    
    homegroup_titles = {'all_homes': 'All homes',
                        'cliff_homes': 'Homes in cliff zones'}
    
    titles = {'stormprob': 'Storm erosion',
            'gradual': 'Gradual erosion',
            'mean_rt': 'Homeowner risk tolerance'}
    
    ylabels = {'mean_rt': 'Mean homeowner risk tolerance',
            'median_value': 'Median home value ($)',
            'n_inaccessible': 'Number of inaccessible homes'}
    
    xlabels = {'gradual': 'Gradual erosion rate (m)',
               'stormprob': 'Storm probability (%)',
               'step': 'Step',
               'mean_rt': 'Mean homeowner risk tolerance'}
    
    ablabels = {'a': 'A',
                'b': 'B',
                'bold': True}

    # file names
    stormprob_runs = ['09102023-1530_0.006m_0pc_2400stp',
                '09102023-1609_0.006m_1pc_2400stp',
                '12102023-0927_0.006m_2pc_2400stp',
                '12102023-1345_0.006m_5pc_2400stp',
                '12102023-1731_0.006m_10pc_2400stp',
                '12102023-2014_0.006m_50pc_2400stp',
                '13102023-1739_0.006m_100pc_2400stp']
    gradualrate_runs = ['09102023-1609_0.006m_1pc_2400stp',
                '14102023-1414_0.01m_1pc_2400stp',
                '14102023-1458_0.016m_1pc_2400stp',
                '14102023-1624_0.041m_1pc_2400stp',
                '14102023-2131_0.083m_1pc_2400stp']
                #'15102023-1423_0.416m_1pc_2400stp']     

    # changes grouping variable
    GROUP_VAR = 'gradual'

    """
    GET DATA AND PLOT
    """
    if GROUP_VAR == 'stormprob':
        files = stormprob_runs
    elif GROUP_VAR == 'gradual':
        files = gradualrate_runs
    elif GROUP_VAR == 'neighbourhood':
        pass

    # load container either by variable or by both variables (erosion type)
    container = parse.load_all_summarised_runs(files, group_by=GROUP_VAR)
    container_all = {
        'gradual': parse.load_all_summarised_runs(gradualrate_runs, group_by='gradual'),
        'stormprob': parse.load_all_summarised_runs(stormprob_runs, group_by='stormprob')}

    def grouped_by_var(save=False):
        fig, ax = bothvars_cliffnoncliff_grouped(container_all, 
                                                 plot_variable='mean_rt', 
                                                 legend_titles=legend_titles, 
                                                 ylabels=ylabels,
                                                 ab_labels=ablabels,
                                                 group_titles=homegroup_titles)
        if save:
            custom_savefig(fig, 'current_plots/botherosion_rt_time.png', 300)
        
        fig, ax = bothvars_cliffnoncliff_grouped(container_all, 
                                                 plot_variable='median_value', 
                                                 legend_titles=legend_titles, 
                                                 ylabels=ylabels,
                                                 ab_labels=ablabels,
                                                 group_titles=homegroup_titles)
        if save:
            custom_savefig(fig, 'current_plots/botherosion_val_time.png', 300)

        fig, ax = total_sales(group_colours=homegroup_colours,
                              home_titles=homegroup_titles,
                              xlabels=xlabels,
                              titles=titles)
        if save:
            custom_savefig(fig, 'current_plots/totalsales_both.png')

        fig, ax = bothvar_cliffnoncliff_individual(data_container=container_all, 
                                            group_titles=homegroup_titles, 
                                            plot_variable='median_value',
                                            ylabels=ylabels,
                                            ab_labels=ablabels,
                                            group_colours=homegroup_colours)
        if save:
            custom_savefig(fig, 'current_plots/tippingpoints_val.png')

        fig, ax = bothvar_cliffnoncliff_individual(data_container=container_all, 
                                            group_titles=homegroup_titles, 
                                            plot_variable='mean_rt',
                                            ylabels=ylabels,
                                            ab_labels=ablabels,
                                            group_colours=homegroup_colours)
        if save:
            custom_savefig(fig, 'current_plots/tippingpoints_rt.png')


        fig, ax = bothvar_cliffnoncliff_individual(data_container=container_all, 
                                                group_titles=homegroup_titles, 
                                                plot_variable='median_value',
                                                ylabels=ylabels,
                                                group_colours=homegroup_colours,
                                                plot_tippingpoints=True,
                                                ab_labels=ablabels)
        if save:
            custom_savefig(fig, 'current_plots/tippingpoints_val_withlines.png')

        fig, ax = bothvar_cliffnoncliff_individual(data_container=container_all, 
                                            group_titles=homegroup_titles, 
                                            plot_variable='mean_rt',
                                            ylabels=ylabels,
                                            group_colours=homegroup_colours,
                                            plot_tippingpoints=True,
                                            ab_labels=ablabels)
        if save:
            custom_savefig(fig, 'current_plots/tippingpoints_rt_withlines.png')

    
    def grouped_plots(save=False):
        
        fig, ax = total_values_both(container_all,
                                    legend_titles=legend_titles,
                                    xlabels=xlabels,
                                    home_titles=homegroup_titles,
                                    colours=homegroup_colours,
                                    ab_labels=ablabels)
        if save:
            custom_savefig(fig, 'current_plots/totalloss_both.png')

        fig, ax = zeroval_rt_kde(plot_type='kde', 
                                 xlabels=xlabels,
                                 legend_titles=legend_titles,
                                 titles=titles)
        if save:
            custom_savefig(fig, 'current_plots/zeroval_kde_both.png')

        """fig, ax = zeroval_rt_kde(plot_type='hist', 
                                 xlabels=xlabels, 
                                 legend_titles=legend_titles,
                                 titles=titles)"""

    # spatial comparisons mainly - focus on the difference between 2 specific model runs
    def individual_run_plots():
        fig, ax = spatial_comparison(run_id='12102023-1731_0.006m_10pc_2400stp',
                                    extent_path='data/input/extent.gpkg',
                                    cliff_region_path='data/input/cliff_region.gpkg',
                                    title=False)
        #custom_savefig(fig, 'current_plots/spatial_006m_10pc.png')
        
        fig, ax = spatial_comparison(run_id='14102023-1624_0.041m_1pc_2400stp',
                                    extent_path='data/input/extent.gpkg',
                                    cliff_region_path='data/input/cliff_region.gpkg',
                                    title=False)
        #custom_savefig(fig, 'current_plots/spatial_041m_1pc.png')
    
    
        fig, ax = geography_changes_stanleypt(runid_1='12102023-1345_0.006m_5pc_2400stp', 
                                            runid_2='12102023-2014_0.006m_50pc_2400stp',
                                            n_arrow_params=(0.05, 0.13, 0.04))
        #custom_savefig(fig, 'current_plots/geogcahnges_stormprob.png')
        
        fig, ax = geography_changes_stanleypt(runid_1='14102023-1414_0.01m_1pc_2400stp', 
                                            runid_2='14102023-1624_0.041m_1pc_2400stp',
                                            n_arrow_params=(0.05, 0.13, 0.04))
        #custom_savefig(fig, 'current_plots/geogchanges_gradualerosion.png')

        fig, ax = final_landmasses(runid_1='14102023-1624_0.041m_1pc_2400stp',
                                runid_2='12102023-2014_0.006m_50pc_2400stp')
    

    def seperate_erosiontypes():
        fig, ax = groupvar_cliffnoncliff_grouped(data_container=container,
                                                plot_variable='mean_rt', 
                                                group_variable=GROUP_VAR, 
                                                legend_titles=legend_titles,
                                                ylabels=ylabels)
        #custom_savefig(fig, 'current_plots/rt_byerosiondist_all.png')
        
        fig, ax = groupvar_cliffnoncliff_grouped(data_container=container,
                                                plot_variable='median_value', 
                                                group_variable=GROUP_VAR, 
                                                legend_titles=legend_titles,
                                                ylabels=ylabels)
        #custom_savefig(fig, 'current_plots/val_byerosiondist_all.png')

        fig, ax = groupvar_cliffnoncliff_grouped(data_container=container,
                                                plot_variable='n_inaccessible', 
                                                group_variable=GROUP_VAR, 
                                                legend_titles=legend_titles,
                                                ylabels=ylabels)
        #custom_savefig(fig, 'current_plots/val_byerosiondist_all.png')

        fig, ax = groupvar_cliffnoncliff_individual(data_container=container, 
                                                    group_variable=GROUP_VAR, 
                                                    titles=titles, 
                                                    plot_variable='mean_rt',
                                                    ylabels=ylabels)
        #custom_savefig(fig, 'current_plots/rt_bystormprob_selected.png')

        fig, ax = groupvar_cliffnoncliff_individual(data_container=container, 
                                                    group_variable=GROUP_VAR, 
                                                    titles=titles, 
                                                    plot_variable='median_value',
                                                    ylabels=ylabels)
        #custom_savefig(fig, 'current_plots/val_byerosiondist_selected.png')

        fig, ax = total_values(data_container=container, 
                           group_variable=GROUP_VAR,
                           legend_titles=legend_titles)
    
        #custom_savefig(fig, 'current_plots/totalloss_byerosiondist.png')

        fig, ax = all_values_grouped(data_container=container, 
                                    group_variable=GROUP_VAR, 
                                    legend_titles=legend_titles, 
                                    titles=titles)
        

    def hex_plots(save=False):
        area = gpd.read_file('data/input/extent.gpkg')
        homes_gradual = pd.read_pickle('output_runs/14102023-1624_0.041m_1pc_2400stp/homes.pkl')
        homes_gradual = homes_gradual.loc[homes_gradual['step'].isin([1, 2400])]
        homes_storm = pd.read_pickle('output_runs/12102023-1731_0.006m_10pc_2400stp/homes.pkl')
        homes_storm = homes_storm.loc[homes_storm['step'].isin([1, 2400])]

        hex = h3fy(area, resolution=11, buffer=False, clip=False)

        gdf_gradual, hex = stats_by_spatialunit_hex(homes_gradual, 1, 2400, 'data/homes_geom_filt40.gpkg',
                                                    hex)
        gdf_storm, _ = stats_by_spatialunit_hex(homes_storm, 1, 2400, 'data/homes_geom_filt40.gpkg',
                                                hex)

        fig, ax = create_hexed_map(area, gdf_gradual, gdf_storm, hex, var='value')
        if save:
            custom_savefig(fig, 'current_plots/hexgrid_val.png', dpi=300)

        fig, ax = create_hexed_map(area, gdf_gradual, gdf_storm, hex, var='owner_risktolerance')
        if save:
            custom_savefig(fig, 'current_plots/hexgrid_rt.png', dpi=300)


    def other_plots():
        fig, ax = income_wealth_example()

        fig, ax = poststormimpact(container,
                                  xlabels=xlabels,
                                  ylabels=ylabels,
                                  titles=titles)
        
        fig, ax = transect_map(show_output=True)

    #individual_run_plots()
    #grouped_by_var(save=True)
    #grouped_plots(save=True)
    #hex_plots(save=True)
    #other_plots()


# %%

# %%
# PLOT: land area

def landarea_experiments():
    all_c_summaries = defaultdict(dict)
    all_c = defaultdict(dict)

    for var in ['gradual', 'stormprob']:
        if var == 'gradual':
            data = gradualrate_runs
        else:
            data = stormprob_runs

        c = defaultdict(dict)
        c_summary = defaultdict(list)

        for x in data:
            file = f'output_runs/{x}/geography.pkl'

            all_geog = pd.read_pickle(file)[1:]
            all_geog = gpd.GeoDataFrame(all_geog, geometry='land_geom')
            all_geog['land_area'] = all_geog.area

            rate, stormprob, steps = vars_from_runid(x)

            if var == 'gradual':
                c[rate] = all_geog
            else:
                c[stormprob] = all_geog

            c_summary['final_area'].append(min(all_geog['land_area']))
            c_summary['gradual_rate'].append(rate)
            c_summary['stormprob'].append(stormprob)

        all_c_summaries[var] = c_summary
        all_c[var] = c

    # PLOT: % of landmass eroded by storm prob and gradual erosion
    original_area = c['09102023-1530_0.006m_0pc_2400stp'].iloc[0]['land_area']

    df_g = pd.DataFrame(all_c_summaries['gradual'])
    df_s = pd.DataFrame(all_c_summaries['stormprob'])

    df_g['perc_area_lost'] = (1 - df_g['final_area'] / original_area) * 100
    df_s['perc_area_lost'] = (1 - df_s['final_area'] / original_area) * 100

    fig, ax = plt.subplots(ncols=2, sharey=True, figsize=(11, 5))

    # change to % of total?
    df_g.plot(x='gradual_rate', y='perc_area_lost', ax=ax[0], marker='.', markersize=12,
            legend=False)
    df_s.plot(x='stormprob', y='perc_area_lost', ax=ax[1], marker='.', markersize=12,
            legend=False)

    ax[0].set(xlabel='Gradual erosion rate (m)', 
            ylabel='Percentage of original landmass area\neroded after model run (%)',
            title='Gradual erosion')
    ax[1].set(xlabel='Storm probability (%)',
            title='Storm erosion')

    #custom_savefig(fig, 'current_plots/landeroded_both.png')


    # PLOT: relationship between land eroded and median value/risk tolerance
    vals = ['1', '10']
    firstyval = 'median_value'
    erosion_var = 'stormprob'

    if firstyval == 'median_value':
        fname = 'value'
    else:
        fname = 'rt'

    fig, ax = plt.subplots(ncols=2, figsize=(16, 6), sharey=True)
    for i, val in enumerate(vals):

        geog_df = all_c[erosion_var][val]
        original_area = geog_df.iloc[0]['land_area']

        df = container['all_homes'][val]
        df['land_area'] = list(geog_df['land_area'])
        df['perc_area_lost'] = (1 - df['land_area'] / original_area) * 100

        for storm_step in list(df.loc[df['storm']]['step']):
            ax[i].axvline(storm_step, linestyle='--', alpha=0.5 if i == 0 else 0.3, color='gray')

        sns.lineplot(df, x='step', y=firstyval, ax=ax[i], color='royalblue')

        if erosion_var == 'median_value':
            plot_title = f'{val} m gradual erosion, baseline storm erosion'
        else:
            plot_title = f'baseline gradual erosion, {val}% storm erosion'
        ax[i].set(ylabel='Median home value ($)' if firstyval == 'median_value' else 'Mean homeowner risk tolerance', 
                xlabel='Step',
                title=plot_title)

        ax2 = ax[i].twinx()
        sns.lineplot(df, x='step', y='perc_area_lost', ax=ax2, color='lightcoral')
        if i == 1:
            ax2.set(ylabel='Percentage of original landmass\narea eroded (%)')
            #ax[i].set_xlim(-75, 1300)
        else:
            ax2.set(ylabel='')
        ax2.set_ylim(-0.2, 3)

        cliff_leg = [Line2D([0], [0], color='royalblue', label='Median home value' if firstyval == 'median_value' else 'Mean homeowner risk tolerance'),
                    Line2D([0], [0], color='lightcoral', label='Percentage of land area eroded'),
                    Line2D([0], [0], color='gray', label='Storm event', linestyle='--')]


    fig.legend(handles=cliff_leg, bbox_to_anchor=(0.6, 0.04))

    #custom_savefig(fig, f'current_plots/{fname}_landarea_{erosion_var}_modifiedsteps.png')
    #custom_savefig(fig, f'current_plots/{fname}_landarea_{erosion_var}.png')



