"""
This file is part of the accompanying code to our paper for reviewing:
"Explaining the mechanism of multiscale groundwater drought events: A new perspective from interpretable deep learning model"

Parts of the script are referred to Jiang, et al. (2021): https://github.com/oreopie/hydro-interpretive-dl.
The code used in this file follows the original license terms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as mpl
import matplotlib.patches as mpatches
import matplotlib.dates as mdates

def plot_droughts(Q, drought_dates, plot_range=[None, None], linecolor="tab:brown", markercolor="tab:red", figsize=(7.5, 2.0)):


    fig, ax = plt.subplots(figsize=figsize)
    fig.tight_layout()

    plot_range[0] = Q.index[0] if plot_range[0] == None else plot_range[0]
    plot_range[1] = Q.index[-1] if plot_range[1] == None else plot_range[1]

    ax.plot(Q["GWS"].loc[plot_range[0]:plot_range[1]], color=linecolor, lw=1.0)
    ax.plot(
        Q.loc[drought_dates, "GWS"].loc[plot_range[0]:plot_range[1]],
        "*",
        c=markercolor,
        markersize=8,
    )

    ax.set_title(f"Identified groundwater droughts from {plot_range[0]} to {plot_range[1]}")
    ax.set_ylabel("GWS(mm)")

    plt.show()



def plot_eg_individual(dataset, drought_eg_dict, drought_eg_var_dict, drought_date, title_suffix=None, linewidth=1.5, figsize=(15, 5)):

    eg_plot = dataset.loc[pd.date_range(end=drought_date, periods=list(drought_eg_dict.values())[0].shape[1]+1, freq='d')[:-1]]

    eg_plot.loc[:, "Rainfall_eg"] = abs(drought_eg_dict[pd.to_datetime(drought_date)][0, :, 0])
    eg_plot.loc[:, "Snowfall_eg"] = abs(drought_eg_dict[pd.to_datetime(drought_date)][0, :, 1])
    eg_plot.loc[:, "tmean_eg"] = abs(drought_eg_dict[pd.to_datetime(drought_date)][0, :, 2])
    eg_plot.loc[:, "ET_eg"] = abs(drought_eg_dict[pd.to_datetime(drought_date)][0, :, 3])
    eg_plot.loc[:, "Canopy_eg"] = abs(drought_eg_dict[pd.to_datetime(drought_date)][0, :, 4])    
    eg_plot.loc[:, "Rainfall_eg_val"] = abs(drought_eg_var_dict[pd.to_datetime(drought_date)][0, :, 0])
    eg_plot.loc[:, "Snowfall_eg_val"] = abs(drought_eg_var_dict[pd.to_datetime(drought_date)][0, :, 1])
    eg_plot.loc[:, "tmean_eg_val"] = abs(drought_eg_var_dict[pd.to_datetime(drought_date)][0, :, 2])
    eg_plot.loc[:, "ET_eg_val"] = abs(drought_eg_var_dict[pd.to_datetime(drought_date)][0, :, 3])
    eg_plot.loc[:, "Canopy_eg_val"] = abs(drought_eg_var_dict[pd.to_datetime(drought_date)][0, :, 4])
  
    #eg_plot.to_csv(r'C:\Users\e0427809\Desktop\GW_Drought\GRACE\Gulf\Timeseries\Output\egplot.csv')
    
    fig = plt.figure(constrained_layout=False, figsize=figsize)

    gs1 = fig.add_gridspec(nrows=4, ncols=1, hspace=0, left=0.00, right=0.45, height_ratios=[2.5,1.5,2.5,1.5])
    ax1 = fig.add_subplot(gs1[0, 0])
    ax2 = fig.add_subplot(gs1[1, 0])
    ax3 = fig.add_subplot(gs1[2, 0])
    ax4 = fig.add_subplot(gs1[3, 0])

    gs2 = fig.add_gridspec(nrows=6, ncols=1, hspace=0, left=0.55, right=1.00, height_ratios=[2.5, 1.5,2.5, 1.5,2.5, 1.5])
    ax5 = fig.add_subplot(gs2[0, 0])
    ax6 = fig.add_subplot(gs2[1, 0])
    ax7 = fig.add_subplot(gs2[2, 0])
    ax8 = fig.add_subplot(gs2[3, 0])
    ax9 = fig.add_subplot(gs2[4, 0])
    ax10 = fig.add_subplot(gs2[5, 0]) 
    

    for ax in [ax1, ax3]:
        ax.spines["bottom"].set_visible(False)
        ax.axes.get_xaxis().set_visible(False)

    for ax in [ax2, ax4]:
        ax.set_ylabel(r'$\phi^{EG}_{i}$')
        ax.spines["top"].set_visible(False)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylim(bottom=np.min(drought_eg_dict[pd.to_datetime(drought_date)]),
                 top=np.max(drought_eg_dict[pd.to_datetime(drought_date)]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))


    ax1.plot(eg_plot['Rainfall'], color='k', lw=linewidth)
    ax1.set_ylabel('Rain [mm/d]', ha='center', y=0.5)

    ax2.plot(eg_plot['Rainfall_eg'], color='blue', lw=linewidth)
    ax2.fill_between(eg_plot['Rainfall_eg'].index,
                     eg_plot['Rainfall_eg']-eg_plot.loc[:, "Rainfall_eg_val"],
                     eg_plot['Rainfall_eg']+eg_plot.loc[:, "Rainfall_eg_val"], color='blue', alpha=0.3)
    ax2.yaxis.label.set_color('blue')
    ax2.tick_params(axis='y', colors='blue')

    ax3.plot(eg_plot['Snowfall'], color='k', lw=linewidth)
    ax3.set_ylabel('Snow [mm/d]', ha='center', y=0.5)

    ax4.plot(eg_plot['Snowfall_eg'], color='blue', lw=linewidth)
    ax4.fill_between(eg_plot['Snowfall_eg'].index,
                     eg_plot['Snowfall_eg']-eg_plot.loc[:, "Snowfall_eg_val"],
                     eg_plot['Snowfall_eg']+eg_plot.loc[:, "Snowfall_eg_val"], color='blue', alpha=0.3)
    ax4.yaxis.label.set_color('blue')
    ax4.tick_params(axis='y', colors='blue')
    
    
    ax5.plot(eg_plot['tmean'], color='k', lw=linewidth)
    ax5.set_ylabel('T [\u2103]', ha='center', y=0.5)

    ax6.plot(eg_plot['tmean_eg'], color='red', lw=linewidth)
    ax6.fill_between(eg_plot['tmean_eg'].index,
                 eg_plot['tmean_eg']-eg_plot.loc[:, "tmean_eg_val"],
                 eg_plot['tmean_eg']+eg_plot.loc[:, "tmean_eg_val"], color='red', alpha=0.3)
    ax6.yaxis.label.set_color('red')
    ax6.tick_params(axis='y', colors='red')
    
    ax7.plot(eg_plot['ET'], color='k', lw=linewidth)
    ax7.set_ylabel('ET', ha='center', y=0.5)    
    
    ax8.plot(eg_plot['ET_eg'], color='red', lw=linewidth)
    ax8.fill_between(eg_plot['ET_eg'].index,
                 eg_plot['ET_eg']-eg_plot.loc[:, "ET_eg_val"],
                 eg_plot['ET_eg']+eg_plot.loc[:, "ET_eg_val"], color='red', alpha=0.3)
    ax8.yaxis.label.set_color('red')
    ax8.tick_params(axis='y', colors='red')    
    
    ax9.plot(eg_plot['Canopy'], color='k', lw=linewidth)
    ax9.set_ylabel('Canopy', ha='center', y=0.5)    
    
    ax10.plot(eg_plot['Canopy_eg'], color='red', lw=linewidth)
    ax10.fill_between(eg_plot['Canopy_eg'].index,
                 eg_plot['Canopy_eg']-eg_plot.loc[:, "Canopy_eg_val"],
                 eg_plot['Canopy_eg']+eg_plot.loc[:, "Canopy_eg_val"], color='red', alpha=0.3)
    ax10.yaxis.label.set_color('red')
    ax10.tick_params(axis='y', colors='red')      
   
    

    ax1.set_title(f"Droughts on {pd.to_datetime(drought_date).strftime('%d %B %Y')} {str(title_suffix)}",
                  fontweight='bold', loc='left')

    plt.show()

def plot_arrow(a1, p1, a2, p2, coordsA='axes fraction', coordsB='axes fraction'):
    con = mpatches.ConnectionPatch(xyA=p1, xyB=p2, coordsA=coordsA, coordsB=coordsB,
                                   axesA=a1, axesB=a2, arrowstyle="-|>", facecolor='black')
    a1.add_artist(con)

def plot_simple_arrow(a1, p1, a2, p2, coordsA='axes fraction', coordsB='axes fraction'):
    con = mpatches.ConnectionPatch(xyA=p1, xyB=p2, coordsA=coordsA, coordsB=coordsB,
                                   axesA=a1, axesB=a2, arrowstyle="->", facecolor='black')
    a1.add_artist(con)

def plot_line(a1, p1, a2, p2, coordsA='axes fraction', coordsB='axes fraction'):
    con = mpatches.ConnectionPatch(xyA=p1, xyB=p2, coordsA=coordsA, coordsB=coordsB,
                                   axesA=a1, axesB=a2)
    a1.add_artist(con)

def plot_decomp(dataset, decomp_dict, drought_date, title_suffix=None, linewidth=1.0, figsize=(15, 8)):

    blue_colors   = mpl.cm.Blues(np.linspace(0,1,16))
    green_colors  = mpl.cm.Greens(np.linspace(0,1,16))
    red_colors    = mpl.cm.Reds(np.linspace(0,1,16))
    purple_colors = mpl.cm.Purples(np.linspace(0,1,16))
    winter_colors = mpl.cm.winter(np.linspace(0,1,16))
    autumn_colors = mpl.cm.autumn(np.linspace(0,1,16))

    decomp_plot = dataset.loc[pd.date_range(end=drought_date, periods=list(decomp_dict.values())[0]['x'].shape[0]+1, freq='d')]

    fig = plt.figure(constrained_layout=False, figsize=figsize)

    gs1 = fig.add_gridspec(nrows=5, ncols=1, hspace=1.2, left=0.000, right=0.180, top=0.70, bottom=0.30)
    gs2 = fig.add_gridspec(nrows=6, ncols=1, hspace=0.6, left=0.250, right=0.550)
    gs3 = fig.add_gridspec(nrows=3, ncols=1, hspace=0.6, left=0.650, right=1.000, top=0.80, bottom=0.20)

    ax1_1 = fig.add_subplot(gs1[0, 0])
    ax1_2 = fig.add_subplot(gs1[1, 0])
    ax1_3 = fig.add_subplot(gs1[2, 0])
    ax1_4 = fig.add_subplot(gs1[3, 0]) 
    ax1_5 = fig.add_subplot(gs1[4, 0])

    ax2_1 = fig.add_subplot(gs2[0, 0])
    ax2_2 = fig.add_subplot(gs2[1, 0])
    ax2_3 = fig.add_subplot(gs2[2, 0])
    ax2_4 = fig.add_subplot(gs2[3, 0])
    ax2_5 = fig.add_subplot(gs2[4, 0])
    ax2_6 = fig.add_subplot(gs2[5, 0])

    ax3_1 = fig.add_subplot(gs3[0, 0])
    ax3_2 = fig.add_subplot(gs3[1, 0])
    ax3_3 = fig.add_subplot(gs3[2, 0])

    ax1_1.plot(decomp_plot['Rainfall'].iloc[:-1], color='k', lw=linewidth)
    ax1_2.plot(decomp_plot['Snowfall'].iloc[:-1], color='k', lw=linewidth)
    ax1_3.plot(decomp_plot['tmean'].iloc[:-1], color='k', lw=linewidth)
    ax1_4.plot(decomp_plot['ET'].iloc[:-1], color='k', lw=linewidth)
    ax1_5.plot(decomp_plot['Canopy'].iloc[:-1], color='k', lw=linewidth)

    
    
    for i in range(16):
        ax2_1.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(drought_date)]['hi_arr'][:, i],
                   c=green_colors[i], alpha=0.60, lw=linewidth)
        ax2_2.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(drought_date)]['hc_arr'][:, i],
                   c=blue_colors[i], alpha=0.60, lw=linewidth)
        ax2_3.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(drought_date)]['hf_arr'][:, i],
                   c=red_colors[i], alpha=0.60, lw=linewidth)
        ax2_4.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(drought_date)]['ho_arr'][:, i],
                   c=purple_colors[i], alpha=0.60, lw=linewidth)
        ax2_5.plot(decomp_plot.index[:], decomp_dict[pd.to_datetime(drought_date)]['c_states'][:, i],
                   c=autumn_colors[i], alpha=0.60, lw=linewidth)
        ax2_6.plot(decomp_plot.index[:], decomp_dict[pd.to_datetime(drought_date)]['h_states'][:, i],
                   c=winter_colors[i], alpha=0.60, lw=linewidth)

        ax3_1.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(drought_date)]['h_update'][:, i],
                   c='#000', alpha=0.60, lw=linewidth*0.6)
        ax3_2.plot(decomp_plot.index[:-1], decomp_dict[pd.to_datetime(drought_date)]['h_forget'][:, i],
                   c='#000', alpha=0.60, lw=linewidth*0.6)

    ax3_3.bar(decomp_plot.index[:-1],
              np.matmul(decomp_dict[pd.to_datetime(drought_date)]['h_forget'][:] * decomp_dict[pd.to_datetime(drought_date)]['h_update'][:],
                        decomp_dict[pd.to_datetime(drought_date)]['dense_W'])[:, 0],
              edgecolor='k',
              width=np.timedelta64(1, 'D'),
              color='red',
              linewidth=0.6)

    ax1_1.set_xticklabels([])
    ax2_1.set_xticklabels([])
    ax2_2.set_xticklabels([])
    ax2_3.set_xticklabels([])
    ax2_4.set_xticklabels([])
    ax2_5.set_xticklabels([])
    ax3_1.set_xticklabels([])
    ax3_2.set_xticklabels([])

    ax1_1.set_title('Rainfall', loc='left', pad=0)  
    ax1_2.set_title('Snowfall',loc='left', pad=0)
    ax1_3.set_title('Temperature', loc='left', pad=0)  
    ax1_4.set_title('Evaportranspiration',loc='left', pad=0)    
    ax1_5.set_title('Plant Canopy', loc='left', pad=0)  
    
    
    ax2_1.set_title(r'Input gate $i_t$', loc='left', pad=0)
    ax2_2.set_title(r'Candidate vector $\tilde{c}_t$', loc='left', pad=0)
    ax2_3.set_title(r'Forget gate $f_t$', loc='left', pad=0)
    ax2_4.set_title(r'Output gate $o_t$', loc='left', pad=0)
    ax2_5.set_title(r'Cell state $c_t$', loc='left', pad=0)
    ax2_6.set_title(r'Hidden state $h_t$', loc='left', pad=0)
    ax3_1.set_title(r'Information initially gained', loc='left', pad=0)
    ax3_2.set_title(r'Proportion to be retained', loc='left', pad=0)
    ax3_3.set_title(r'Information actually contributed', loc='left', pad=0)

    ax1_2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1_2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for tick in ax1_2.xaxis.get_ticklabels()[1:3] + ax1_2.xaxis.get_ticklabels()[4:6]:
        tick.set_visible(False)

    ax2_6.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax2_6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for tick in ax2_6.xaxis.get_ticklabels()[1::2]:
        tick.set_visible(False)

    ax3_3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax3_3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for tick in ax3_3.xaxis.get_ticklabels()[1::2]:
        tick.set_visible(False)

    ax3_2.set_ylim(bottom=-0.1*np.percentile(decomp_dict[pd.to_datetime(drought_date)]['h_forget'][-1, :], q=0.75),
                   top=np.percentile(decomp_dict[pd.to_datetime(drought_date)]['h_forget'][-1, :], q=0.75))


    plot_simple_arrow(ax2_3, (1.01, 0.5), ax3_1, (-0.15, 0.60))
    plot_simple_arrow(ax2_4, (1.01, 0.5), ax3_1, (-0.15, 0.50))
    plot_simple_arrow(ax2_6, (1.01, 0.5), ax3_1, (-0.15, 0.40))

    plot_simple_arrow(ax2_3, (1.01, 0.5), ax3_2, (-0.10, 0.525))
    plot_simple_arrow(ax2_4, (1.01, 0.5), ax3_2, (-0.10, 0.475))

    plot_line(ax3_1, (1.02, 0.5), ax3_1, (1.08, 0.5))
    plot_line(ax3_2, (1.02, 0.5), ax3_2, (1.08, 0.5))
    plot_line(ax3_3, (1.02, 0.5), ax3_3, (1.08, 0.5))
    plot_line(ax3_1, (1.08, 0.5), ax3_3, (1.08, 0.5))
    plot_arrow(ax3_3, (1.08, 0.5), ax3_3, (1.005, 0.5))
    ax3_2.annotate(r'$\bigodot$', (1.05, -0.4), xycoords='axes fraction', backgroundcolor='white')

    fig.suptitle(f"Drought on {pd.to_datetime(drought_date).strftime('%d %B %Y')} {str(title_suffix)}",
                 fontweight='bold', x=0, ha='left')

    plt.show()
