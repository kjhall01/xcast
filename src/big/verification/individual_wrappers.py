import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def index_of_agreement_(x, y):
	"""implements index of agreement metric"""
	return 1 - np.sum((y-x)**2) / np.sum(  (np.abs(y - np.nanmean(x)) + np.abs(x - np.nanmean(x)))**2)

def spearman_coef_(x, y):
	"""abstracts out coefficient from stats.spearmanr"""
	coef, p = stats.spearmanr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return coef

def spearman_p_(x, y):
	"""abstracts out p from stats.spearmanr"""
	coef, p = stats.spearmanr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return p

def pearson_coef_(x, y):
	"""abstracts out coefficient from stats.pearsonr"""
	coef, p = stats.pearsonr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return coef

def pearson_p_(x, y):
	"""abstracts out p from stats.pearsonr"""
	coef, p = stats.pearsonr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return p

def root_mean_squared_error_(x, y):
	"""abstracts out rmse from sklearn.metrics.mean_squared_error"""
	return mean_squared_error(np.squeeze(y), np.squeeze(x), squared=False)

def mean_squared_error_(x, y):
	"""abstracts out mse from sklearn.metrics.mean_squared_error"""
	return mean_squared_error(np.squeeze(y), np.squeeze(x), squared=True)

def mean_absolute_error_(x, y):
	"""abstracts out mae from sklearn.metrics.mean_absolute_error"""
	return mean_absolute_error(np.squeeze(y), np.squeeze(x))
