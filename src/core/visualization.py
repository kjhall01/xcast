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
from scipy import interp
from itertools import cycle

import dask.array as da
import xarray as xr
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from scipy import stats

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


def view_probabilistic(X, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='T', sample_ndx=0, nanmask=None):
	"""Plots the max value out of a three-category data array at each point across time and space. uses cartopy to plot maps on top. """
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

def view_roc(X, Y, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='T', y_lat_dim='Y', y_lon_dim='X', y_feature_dim='M', y_sample_dim='T' ):
	"""where X is predicted, and Y is observed"""
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
	x_data = X1.values.reshape(len(X1.coords[x_lat_dim].values)*len(X1.coords[x_lon_dim].values)*len(X1.coords[x_sample_dim].values), len(X1.coords[x_feature_dim].values))
	y_data = Y1.values.reshape(len(Y1.coords[y_lat_dim].values)*len(Y1.coords[y_lon_dim].values)*len(Y1.coords[y_sample_dim].values), len(Y1.coords[y_feature_dim].values))
	tst = x_data *y_data
	x_data = x_data[~np.isnan(tst).any(axis=1)]
	y_data = y_data[~np.isnan(tst).any(axis=1)]
	n_classes = len(X1.coords[x_feature_dim].values)
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_data[:, i], x_data[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# First aggregate all false positive rates
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
		mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# Plot all ROC curves
	plt.figure()
	plt.plot(fpr["macro"], tpr["macro"],
				label='macro-average ROC curve (area = {0:0.2f})'
				''.format(roc_auc["macro"]),
				color='navy', linestyle=':', linewidth=4)

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=2,
				label='ROC curve of class {0} (area = {1:0.2f})'
				''.format(X1.coords[x_feature_dim].values[i], roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=2)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.show()


def view_reliability(X, Y, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='T', y_lat_dim='Y', y_lon_dim='X', y_feature_dim='M', y_sample_dim='T' ):
	"""where X is predicted, and Y is observed"""
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
	x_data = X1.values.reshape(len(X1.coords[x_lat_dim].values)*len(X1.coords[x_lon_dim].values)*len(X1.coords[x_sample_dim].values), len(X1.coords[x_feature_dim].values))
	y_data = Y1.values.reshape(len(Y1.coords[y_lat_dim].values)*len(Y1.coords[y_lon_dim].values)*len(Y1.coords[y_sample_dim].values), len(Y1.coords[y_feature_dim].values))
	tst = x_data *y_data
	x_data = x_data[~np.isnan(tst).any(axis=1)]
	y_data = y_data[~np.isnan(tst).any(axis=1)]
	n_classes = len(X1.coords[x_feature_dim].values)
	fpr = dict()
	tpr = dict()
	for i in range(n_classes):
		fpr[i], tpr[i] = calibration_curve(y_data[:, i], x_data[:, i])

	# Plot all ROC curves
	plt.figure()
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=2,
				label='Reliability of Class {0}'
				''.format(X1.coords[x_feature_dim].values[i]))

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Mean Predicted Probability')
	plt.ylabel('Fraction of Positives')
	plt.title('Reliability Diagram')
	plt.legend(loc="lower right")
	plt.show()


def view_taylor(X, Y, x_lat_dim='Y', x_lon_dim='X', x_feature_dim='M', x_sample_dim='T', y_lat_dim='Y', y_lon_dim='X', y_feature_dim='M', y_sample_dim='T' ):
	"""where X is predicted, and Y is observed"""
	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
	x_data = X1.values.reshape(len(X1.coords[x_lat_dim].values)*len(X1.coords[x_lon_dim].values)*len(X1.coords[x_sample_dim].values), len(X1.coords[x_feature_dim].values))
	y_data = Y1.values.reshape(len(Y1.coords[y_lat_dim].values)*len(Y1.coords[y_lon_dim].values)*len(Y1.coords[y_sample_dim].values), len(Y1.coords[y_feature_dim].values))
	tst = x_data *y_data
	x_data = x_data[~np.isnan(tst).any(axis=1)]
	y_data = y_data[~np.isnan(tst).any(axis=1)]
	n_classes = len(X1.coords[x_feature_dim].values)

	obs_stddev = y_data.std()
	stddevs = x_data.std(axis=0)

	correlations = []
	for i in range(n_classes):
		try:
			coef, p = stats.pearsonr(np.squeeze(x_data[:,i]).astype(float), np.squeeze(y_data[:,0]).astype(float))
			correlations.append(coef)
		except:
			correlations.append(np.nan)
	obs_cor = 1.0
	correlations = np.asarray(correlations)

	obs_rmsd = 0
	rmsds = np.sqrt(obs_stddev**2 + stddevs**2 - 2* obs_stddev*stddevs*correlations)

	angles = (1 - correlations ) * np.pi / 2.0

	xs = [np.cos(angles[i]) * stddevs[i] for i in range(stddevs.shape[0])]
	ys = [np.sin(angles[i]) * stddevs[i] for i in range(stddevs.shape[0])]

	fig = plt.figure(frameon=False, figsize=(5,5))
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'purple', 'green'])
	for i, color in zip(range(len(xs)), colors):
		plt.scatter(xs[i], ys[i], color=color, lw=2, label='Model {}'.format(X1.coords[x_feature_dim].values[i]))
	plt.scatter(obs_stddev, 0, color='red', label='Observations')

	for i in range(4):
		circle1 = plt.Circle((obs_stddev, 0), max(rmsds)*((i+1) / 4.0), edgecolor='green', fill=False, alpha=0.5, linestyle='-.')
		fig.axes[0].add_patch(circle1)
		fig.axes[0].annotate('{:>02.2}'.format(max(rmsds)*((i+1) / 4.0)), (obs_stddev, max(rmsds)*((i+1.1) / 4.0)), (obs_stddev, max(rmsds)*((i+1.1) / 4.0)), color='green', alpha=0.5, size=8)

	fig.axes[0].annotate('RMS', (obs_stddev, max(rmsds)*1.1), (obs_stddev, max(rmsds)*1.1), color='green', alpha=0.5)


	for i in range(7):
		circle1 = plt.Circle((0, 0), obs_stddev*(i / 3.0), edgecolor='black', fill=False)
		fig.axes[0].add_patch(circle1)


	for i in range(5):
		angle = np.pi / 2.0 * (1 - (i+0.5)/5.0)
		plt.plot([0, np.cos(angle)*obs_stddev*1.5], [0, np.sin(angle)*obs_stddev*1.5], linewidth=0.5, color='blue', alpha=0.5, linestyle='-.')
		fig.axes[0].annotate('{}'.format((i+0.5) / 5.0), (np.cos(angle)*obs_stddev*1.5, np.sin(angle)*obs_stddev*1.5), (np.cos(angle)*obs_stddev*1.5, np.sin(angle)*obs_stddev*1.5), alpha=0.5, color='blue', size=8, rotation=(1 - (i+0.5)/5.0)*90)
	fig.axes[0].annotate('Peason Correlation', (np.cos(np.pi/4)*obs_stddev*1.45, np.sin(np.pi/4)*obs_stddev*1.45), (np.cos(np.pi/4)*obs_stddev*1.45, np.sin(np.pi/4)*obs_stddev*1.45), color='blue', rotation=315, alpha=0.5)

	plt.xlim([0,  obs_stddev * 1.6])
	plt.ylim([0,  obs_stddev * 1.6])
	plt.xticks([obs_stddev * i / 3.0 for i in range(5)], ['{:<02.2}'.format(obs_stddev * i / 3.0) for i in range(5)])
	plt.yticks([obs_stddev * i / 3.0 for i in range(5)], ['{:<02.2}'.format(obs_stddev * i / 3.0) for i in range(5)])
	plt.xlabel('Standard Deviation')
	plt.ylabel('Standard Deviation')
	plt.title('Taylor Diagram')
	plt.legend(loc="lower left")
	plt.show()
