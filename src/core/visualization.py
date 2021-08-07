import matplotlib.pyplot as plt
import numpy as np
import warnings
import datetime as dt
from .utilities import *
import cartopy.crs as ccrs
from cartopy import feature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as colors
import matplotlib as mpl
import copy

class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		midpoint = midpoint
		self.vmin, self.vmax, self.midpoint = vmin, vmax, midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))

def view_probabilistic(X, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='T', sample_ndx=0, nanmask=None):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	dc = {x_sample_dim: sample_ndx}
	X1 = X.isel(**dc).transpose(x_lat_dim, x_lon_dim, x_feature_dim)
	x_data = copy.deepcopy(X1.values)
	maxs  = np.argmax(x_data, axis=-1)
	x_data[:, :, 0][np.where(maxs != 0)] = np.nan
	x_data[:, :, 1][np.where(maxs != 1)] = np.nan
	x_data[:, :, 2][np.where(maxs != 2)] = np.nan
	fig, ax = plt.subplots(nrows=1, ncols=1,  sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
	states_provinces = feature.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', facecolor='none')
	ax.set_extent([np.min(X1.coords[x_lon_dim].values),np.max(X1.coords[x_lon_dim].values), np.min(X1.coords[x_lat_dim].values), np.max(X1.coords[x_lat_dim].values)], ccrs.PlateCarree()) #sets the lat/long boundaries of the plot area
	pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4))) #adds dotted gridlines to plot
	pl.right_labels, pl.top_labels, pl.bottom_labels,  pl.left_labels  = True, False , True, False #adds labels to dashed gridlines on left and bottom
	pl.xformatter, pl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER #sets formatters - arcane cartopy stuff
	ax.add_feature(states_provinces, edgecolor='black') #adds the cartopy default map to the plot

	#fancy probabilistic
	grays =  mpl.cm.Greys(np.linspace(0,1,20))
	grays = mpl.colors.ListedColormap(grays[:5,:-1])
	CS1 = ax.pcolormesh(X1.coords[x_lon_dim].values, X1.coords[x_lat_dim].values, x_data[:, :, 0]*100,vmin=35, vmax=85, cmap=plt.get_cmap('YlOrBr', 5))
	CS2 = ax.pcolormesh(X1.coords[x_lon_dim].values, X1.coords[x_lat_dim].values, x_data[:, :, 1]*100, vmin=35, vmax=55, cmap=grays)
	CS3 = ax.pcolormesh(X1.coords[x_lon_dim].values, X1.coords[x_lat_dim].values, x_data[:, :, 2]*100, vmin=35, vmax=85, cmap=plt.get_cmap('GnBu', 5))

	bounds = [40,50,60,70, 80]
	nbounds = [40, 45, 50]

	#fancy probabilistic cb bottom
	axins_f_bottom = inset_axes(ax, width="40%",  height="5%", loc='lower left', bbox_to_anchor=(-0.2, -0.15, 1.2, 1), bbox_transform=ax.transAxes, borderpad=0.1 )
	axins2_bottom = inset_axes(ax, width="20%",   height="5%",  loc='lower center', bbox_to_anchor=(-0.0, -0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0.1 )
	axins3_bottom = inset_axes(ax, width="40%",   height="5%",  loc='lower right', bbox_to_anchor=(0, -0.15, 1.2, 1), bbox_transform=ax.transAxes, borderpad=0.1 )
	cbar_fbl = fig.colorbar(CS1, ax=ax, cax=axins_f_bottom, orientation='horizontal', ticks=bounds)
	cbar_fbl.set_label('BN Probability (%)') #, rotation=270)\

	cbar_fbc = fig.colorbar(CS2, ax=ax,  cax=axins2_bottom, orientation='horizontal', ticks=nbounds)
	cbar_fbc.set_label('N Probability (%)') #, rotation=270)\

	cbar_fbr = fig.colorbar(CS3, ax=ax,  cax=axins3_bottom, orientation='horizontal', ticks=bounds)
	cbar_fbr.set_label('AN Probability (%)') #, rotation=270)\
	plt.show()


def view_deterministic(X, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='T', sample_ndx=0, nanmask=None, vmin=None, vmax=None):
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	dc = {x_sample_dim: sample_ndx}
	X1 = X.isel(**dc).transpose(x_lat_dim, x_lon_dim, x_feature_dim)
	x_data = copy.deepcopy(X1.values)

	fig, ax = plt.subplots(nrows=1, ncols=1,  sharex=False,sharey=False, subplot_kw={'projection': ccrs.PlateCarree()})
	states_provinces = feature.NaturalEarthFeature(category='cultural', name='admin_0_countries', scale='10m', facecolor='none')
	ax.set_extent([np.min(X1.coords[x_lon_dim].values),np.max(X1.coords[x_lon_dim].values), np.min(X1.coords[x_lat_dim].values), np.max(X1.coords[x_lat_dim].values)], ccrs.PlateCarree()) #sets the lat/long boundaries of the plot area
	pl=ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle=(0,(2,4))) #adds dotted gridlines to plot
	pl.right_labels, pl.top_labels, pl.bottom_labels,  pl.left_labels  = True, False , True, False #adds labels to dashed gridlines on left and bottom
	pl.xformatter, pl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER #sets formatters - arcane cartopy stuff
	ax.add_feature(states_provinces, edgecolor='black') #adds the cartopy default map to the plot

	#fancy probabilistic
	if vmin is not None and vmax is not None:
		CS1 = ax.pcolormesh(X1.coords[x_lon_dim].values, X1.coords[x_lat_dim].values, x_data[:, :, 0], vmin=vmin, vmax=vmax, cmap=plt.get_cmap('BrBG', 10))
		ticks = [vmin +(vmax-vmin)*0.05 + (vmax-vmin)*0.1 * i for i in range(10)]
	else:
		CS1 = ax.pcolormesh(X1.coords[x_lon_dim].values, X1.coords[x_lat_dim].values, x_data[:, :, 0], cmap=plt.get_cmap('BrBG', 10))

	#fancy probabilistic cb bottom
	axins_f_bottom = inset_axes(ax, width="100%",  height="5%", loc='lower center', bbox_to_anchor=(-0.0, -0.15, 1.0, 1), bbox_transform=ax.transAxes, borderpad=0.1 )
	if vmin is not None and vmax is not None:
		cbar_fbl = fig.colorbar(CS1, ax=ax, cax=axins_f_bottom, orientation='horizontal', ticks=ticks)
	else:
		cbar_fbl = fig.colorbar(CS1, ax=ax, cax=axins_f_bottom, orientation='horizontal')
	cbar_fbl.set_label('Variable (units)') #, rotation=270)\
	plt.show()
