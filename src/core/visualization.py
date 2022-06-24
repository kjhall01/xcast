import matplotlib.pyplot as plt
import numpy as np
import warnings
import datetime as dt
from .utilities import *

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as colors
import matplotlib as mpl
import copy
import uuid
from pathlib import Path
from scipy import interp
from itertools import cycle

import dask.array as da
import xarray as xr
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.ndimage import gaussian_filter

def __gaussian_smooth(X, x_lat_dim='Y', x_lon_dim='X', x_sample_dim='T', x_feature_dim='M', kernel=(9,9), use_dask=False, feature_chunks=1, sample_chunks=1, destination='.xcast_worker_space' ):
	check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
	#X1 = fill_space_mean(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim )
	X1 = X.chunk({x_feature_dim: max(X.shape[list(X.dims).index(x_feature_dim)] // feature_chunks, 1), x_sample_dim: max(X.shape[list(X.dims).index(x_sample_dim)] // sample_chunks, 1) }).transpose(x_feature_dim, x_sample_dim, x_lat_dim, x_lon_dim)

	hdf=None
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
		results[i] = np.concatenate(results[i], axis=1)
	results = np.concatenate(results, axis=0)
	attrs = X1.attrs({'generated by': 'XCast Gaussian Smoothing {}'.format(kernel)})
	return xr.DataArray(data=results, coords=X1.coords, dims=X1.dims, attrs=attrs)


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
			blurred = gaussian_filter(toblur2, sigma=kernel)
			blurred[mask] = np.nan
			res[i].append(blurred)
	res = np.asarray(res)
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


def view_roc(X, Y, x_lat_dim=None, x_lon_dim=None, x_feature_dim=None, x_sample_dim=None, y_lat_dim=None, y_lon_dim=None, y_feature_dim=None, y_sample_dim=None ):
	"""where X is predicted, and Y is observed"""
	x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	#X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	#Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
	#x_data = X1.values.reshape(len(X1.coords[x_lat_dim].values)*len(X1.coords[x_lon_dim].values)*len(X1.coords[x_sample_dim].values), len(X1.coords[x_feature_dim].values))
	#y_data = Y1.values.reshape(len(Y1.coords[y_lat_dim].values)*len(Y1.coords[y_lon_dim].values)*len(Y1.coords[y_sample_dim].values), len(Y1.coords[y_feature_dim].values))
	x_data = X.stack(point=(x_lat_dim, x_lon_dim, x_sample_dim)).transpose('point', x_feature_dim).values 
	y_data = Y.stack(point=(y_lat_dim, y_lon_dim, y_sample_dim)).transpose('point', y_feature_dim).values 

	
	tst = x_data *y_data
	x_data = x_data[~np.isnan(tst).any(axis=1)]
	y_data = y_data[~np.isnan(tst).any(axis=1)]
	n_classes = len(X.coords[x_feature_dim].values)
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
				''.format(X.coords[x_feature_dim].values[i], roc_auc[i]))

	plt.plot([0, 1], [0, 1], 'k--', lw=2)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve')
	plt.legend(loc="lower right")
	plt.show()


def view_reliability(X, Y, x_lat_dim=None, x_lon_dim=None, x_feature_dim=None, x_sample_dim=None, y_lat_dim=None, y_lon_dim=None, y_feature_dim=None, y_sample_dim=None ):
	"""where X is predicted, and Y is observed"""
	x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	#X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	#Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
	#x_data = X1.values.reshape(len(X1.coords[x_lat_dim].values)*len(X1.coords[x_lon_dim].values)*len(X1.coords[x_sample_dim].values), len(X1.coords[x_feature_dim].values))
	#y_data = Y1.values.reshape(len(Y1.coords[y_lat_dim].values)*len(Y1.coords[y_lon_dim].values)*len(Y1.coords[y_sample_dim].values), len(Y1.coords[y_feature_dim].values))
	
	x_data = X.stack(point=(x_lat_dim, x_lon_dim, x_sample_dim)).transpose('point', x_feature_dim).values 
	y_data = Y.stack(point=(y_lat_dim, y_lon_dim, y_sample_dim)).transpose('point', y_feature_dim).values 
	tst = x_data * y_data
	x_data = x_data[~np.isnan(tst).any(axis=1)]
	y_data = y_data[~np.isnan(tst).any(axis=1)]

	n_classes = len(X.coords[x_feature_dim].values)
	fpr = dict()
	tpr = dict()
	for i in range(n_classes):
		fpr[i], tpr[i] = calibration_curve(y_data[:, i], x_data[:, i], strategy='quantile')

	# Plot all ROC curves
	plt.figure()
	colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color, lw=2,
				label='Reliability of Class {0}'
				''.format(X.coords[x_feature_dim].values[i]))

	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Mean Predicted Probability')
	plt.ylabel('Fraction of Positives')
	plt.title('Reliability Diagram')
	plt.legend(loc="lower right")
	plt.show()


def view_taylor(X, Y, x_lat_dim=None, x_lon_dim=None, x_feature_dim=None, x_sample_dim=None, y_lat_dim=None, y_lon_dim=None, y_feature_dim=None, y_sample_dim=None ):
	"""where X is predicted, and Y is observed"""
	x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

	X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
	Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
	#x_data = X1.values.reshape(len(X1.coords[x_lat_dim].values)*len(X1.coords[x_lon_dim].values)*len(X1.coords[x_sample_dim].values), len(X1.coords[x_feature_dim].values))
	#y_data = Y1.values.reshape(len(Y1.coords[y_lat_dim].values)*len(Y1.coords[y_lon_dim].values)*len(Y1.coords[y_sample_dim].values), len(Y1.coords[y_feature_dim].values))
	x_data = X.stack(point=(x_lat_dim, x_lon_dim, x_sample_dim)).transpose('point', x_feature_dim).values 
	y_data = Y.stack(point=(y_lat_dim, y_lon_dim, y_sample_dim)).transpose('point', y_feature_dim).values 

	
	tst = x_data *y_data
	x_data = x_data[~np.isnan(tst).any(axis=1)]
	y_data = y_data[~np.isnan(tst).any(axis=1)]
	n_classes = len(X.coords[x_feature_dim].values)

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
		plt.scatter(xs[i], ys[i], color=color, lw=2, label='Model {}'.format(X.coords[x_feature_dim].values[i]))
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
