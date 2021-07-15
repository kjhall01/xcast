import matplotlib.pyplot as plt
import numpy as np
import warnings
import datetime as dt

import cartopy.crs as ccrs
from cartopy import feature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def multimap(args, coords):
	for arg in args:
		assert len(arg.values.shape) == 2, '{} must pass 2D data for multimap'.format(dt.datetime.now())
	N_MMES = len(args)
	assert len(args) == len(coords), "{} Must pass same number of args and coords".format(dt.datetime.now())
	fig, ax = plt.subplots(nrows=N_MMES, ncols=1, figsize=(4*N_MMES,4), sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()}) #creates pyplot plotgrid with maps
	if N_MMES == 1:
		ax = [ax]
	else:
		pass

	states_provinces = feature.NaturalEarthFeature(category='cultural', name='admin_0_countries',scale='10m',facecolor='none')#setting more variables
	for i in range(N_MMES): #adjustable
		ax[i].set_extent([np.min(args[i].coords[coords[i]['X']].values),np.max(args[i].coords[coords[i]['X']].values), np.min(args[i].coords[coords[i]['Y']].values), np.max(args[i].coords[coords[i]['Y']].values)], ccrs.PlateCarree()) #sets the lat/long boundaries of the plot area
		ax[i].add_feature(feature.LAND) #adds predefined cartopy land feature - gets overwritten
		pl=ax[i].gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4))) #adds dotted gridlines to plot
		pl.right_labels, pl.top_labels, pl.bottom_labels,  pl.left_labels  = True, False , True, False #adds labels to dashed gridlines on left and bottom
		pl.xformatter, pl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER #sets formatters - arcane cartopy stuff

		ax[i].add_feature(states_provinces, edgecolor='black') #adds the cartopy default map to the plot


		ax[i].text(-0.25, 0.5, '{}'.format(args[i].name),rotation='vertical', verticalalignment='center', horizontalalignment='center', transform=ax[i].transAxes) #print title vertially on the left side

		var = args[i].values
		if args[i].name in ['spearman_coeff', 'pearson_coeff']:
			CS1 = ax[i].pcolormesh(args[i].coords[coords[i]['X']].values, args[i].coords[coords[i]['Y']].values, var, vmin=-1, vmax=1, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else
		elif args[i].name in ['index_of_agreement']:
			CS1 = ax[i].pcolormesh(args[i].coords[coords[i]['X']].values, args[i].coords[coords[i]['Y']].values, var, vmin=0, vmax=1, cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else
		elif args[i].name in ['root_mean_squared_error']:
			CS1 = ax[i].pcolormesh(args[i].coords[coords[i]['X']].values, args[i].coords[coords[i]['Y']].values, var, vmin=-1, vmax=1, cmap='Reds') #adds probability of below normal where below normal is most likely  and nan everywhere else
		else:
			CS1 = ax[i].pcolormesh(args[i].coords[coords[i]['X']].values, args[i].coords[coords[i]['Y']].values, var,  cmap='RdYlBu') #adds probability of below normal where below normal is most likely  and nan everywhere else


			axins = inset_axes(ax[i], width="100%", height="5%",  loc='lower center', bbox_to_anchor=(0., -0.2, 1, 1), bbox_transform=ax[i].transAxes, borderpad=0.15,) #describes where colorbar should go
			cbar_bdet = fig.colorbar(CS1, ax=ax[i],  cax=axins, orientation='horizontal', pad = 0.02) #add colorbar based on hindcast data
	plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)
	plt.show()
