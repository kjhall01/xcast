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
import cv2, uuid, h5py
from pathlib import Path
import dask.array as da
import xarray as xr


def __gaussian_smooth(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, feature_chunks=1, sample_chunks=1, destination='.xcast_worker_space' ):
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
	#X1 = fill_space_mean(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim )
	X1 = X.chunk({x_feature_dim: max(X.shape[list(X.dims).index(x_feature_dim)] // feature_chunks, 1), x_sample_dim: max(X.shape[list(X.dims).index(x_sample_dim)] // sample_chunks, 1) }).transpose(x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)

	if use_dask:
		id =  Path(destination) / str(uuid.uuid4())
		hdf = h5py.File(id, 'w')
	else:
		hdf = None

	results, seldct = [], {}
	feature_ndx = 0
	for i in range(len(X1.chunks[list(X1.dims).index(x_feature_dim)])):
		sample_ndx = 0
		results.append([])
		for j in range(len(X1.chunks[list(X1.dims).index(x_sample_dim)])):
			x_isel = {x_feature_dim: slice(feature_ndx, feature_ndx + X1.chunks[list(X1.dims).index(x_feature_dim)][i]), x_sample_dim: slice(sample_ndx, sample_ndx + X1.chunks[list(X1.dims).index(x_sample_dim)][j])}
			results[i].append(__gaussian_smooth_chunk(X1.isel(**x_isel), feature_ndx, sample_ndx, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim , use_dask=use_dask, kernel=kernel, hdf=hdf))
			sample_ndx += X1.chunks[list(X1.dims).index(x_sample_dim)][j]
		feature_ndx += X1.chunks[list(X1.dims).index(x_feature_dim)][i]
		if not use_dask:
			results[i] = np.concatenate(results[i], axis=1)
	if not use_dask:
		results = np.concatenate(results, axis=0)
	else:
		results = []
		hdf.close()
		hdf = h5py.File(id, 'r')
		feature_ndx = 0
		for i in range(len(X1.chunks[list(X1.dims).index(x_feature_dim)])):
			sample_ndx = 0
			results.append([])
			for j in range(len(X1.chunks[list(X1.dims).index(x_sample_dim)])):
				results[i].append(da.from_array(hdf['data_{}_{}'.format(feature_ndx, sample_ndx)]))
				sample_ndx += X1.chunks[list(X1.dims).index(x_sample_dim)][j]
			results[i] = da.concatenate(results[i], axis=0)
			feature_ndx += X1.chunks[list(X1.dims).index(x_feature_dim)][i]
		results = da.concatenate(results, axis=1)
	#X1 = X1.transpose(x_sample_dim, x_feature_dim, x_lat_dim, x_lon_dim)
	return xr.DataArray(data=results, coords=X1.coords, dims=X1.dims)


def __gaussian_smooth_chunk(X, feature_ndx, sample_ndx, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, hdf=None ):
	res = []
	data = X.values
	for i in range(data.shape[0]):
		res.append([])
		for j in range(data.shape[1]):
			toblur = data[i, j, :, :]
			mask = np.isnan(toblur)
			toblur2 = toblur.copy()
			toblur2[mask] = np.nanmean(toblur)
			blurred = cv2.GaussianBlur(toblur2, kernel,0)
			blurred[mask] = np.nan
			res[i].append(blurred)
	res = np.asarray(res)
	if use_dask:
		hdf.create_dataset('data_{}_{}'.format(feature_ndx, sample_ndx), data=res)
		return 'data_{}_{}'.format(feature_ndx, sample_ndx)
	else:
		return res




class MidpointNormalize(colors.Normalize):
	"""Helper class for setting the midpoint of a pyplot colorbar"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		midpoint = midpoint
		self.vmin, self.vmax, self.midpoint = vmin, vmax, midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))


def view_deterministic(X, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='T', sample_ndx=0, nanmask=None, vmin=None, vmax=None):
	"""plots value of xarray dataset at a given sample ndx , uses cartopy to plot maps"""
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
	return fig

def view_skill(skill, opfile=None, blur=None):

	plots = [[skill.symmetric_mean_absolute_percentage_error, skill.root_mean_squared_error, skill.median_absolute_error, skill.mean_squared_error],
			[skill.mean_error, skill.mean_absolute_percentage_error, skill.mean_absolute_error, skill.determination_coefficient],
			[skill.spearman_p_value, skill.spearman_effective_p_value, skill.slope_linear_fit, skill.pearson_p_value],
			[skill.pearson_effective_p_value, skill.effective_sample_size, skill.spearman_coefficient, skill.pearson_coefficient]]
	names = [[i.name for i in plots[j]] for j in range(len(plots))]
	if blur is not None:
		assert blur[0] % 2 == 1 and blur[1] % 2 == 1, 'invalid gaussian_smoothing kernel {}'.format((blur[0], blur[1]))
		kernel = (blur[0], blur[1])
		for i in range(len(plots)):
			for j in range(len(plots[i])):
				plots[i][j] = __gaussian_smooth(plots[i][j].expand_dims({'T':[0]}), x_lat_dim='lat', x_lon_dim='lon', x_sample_dim='T', x_feature_dim='member', kernel=kernel, use_dask=True).isel(member=0, T=0)

	fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, subplot_kw={'projection':ccrs.PlateCarree()}, figsize=(28, 24))
	for i in range(len(plots)):
		for j in range(len(plots[0])):
			plots[i][j].plot(ax=axes[i][j], cmap='RdBu')
			axes[i][j].set_title(names[i][j])

			axes[i][j].coastlines()
			gl = axes[i][j].gridlines()
			gl.xlabels_bottom, gl.ylabels_left = True, True
	if opfile is not None:
		fig.savefig(opfile, dpi=300)
	plt.show()
