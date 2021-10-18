import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from .base_verification import *


def _LogLikelihood(x, y):
	n, k =  x.shape
	residuals = np.squeeze(x - y)
	return -(n * 1/2) * (1 + np.log(2 * np.pi)) - (n / 2) * np.log(residuals.dot(residuals) / n)

def _aic(x, y):
	ll = _LogLikelihood(x, y)
	n, k = x.shape
	return (-2 * ll) + (2 * k)

@metric
def bayesian_information_criterion(x, y):
	ll = _LogLikelihood(x, y)
	n, k = x.shape
	return (-2 * ll) + (k * np.log(n))

@metric
def akaike_information_criterion(x, y):
	ll = _LogLikelihood(x, y)
	n, k = x.shape
	return (-2 * ll) + (2 * k)

@metric
def log_likelihood(x, y):
	n, k =  x.shape
	residuals = np.squeeze(x - y)
	return -(n * 1/2) * (1 + np.log(2 * np.pi)) - (n / 2) * np.log(residuals.dot(residuals) / n)

@metric
def roc_auc(x, y):
	try:
		ret =  roc_auc_score(x, y, average=None, multi_class='ovr', labels=[0,1,2])
		return ret
	except:
		return np.asarray([0,0,0])

@metric
def f1(x, y):
	try:
		ret =  f1_score(np.argmax(y, axis=-1), np.argmax(x, axis=-1), average=None, labels=[0,1,2])
		return ret
	except:
		return np.asarray([0,0,0])

@metric
def average_precision(x, y):
	return average_precision_score(x, y, average=None)

@metric
def index_of_agreement(x, y):
	"""implements index of agreement metric"""
	return 1 - np.sum((y-x)**2) / np.sum(  (np.abs(y - np.nanmean(x)) + np.abs(x - np.nanmean(x)))**2)

@metric
def nash_sutcliffe_efficiency(x, y):
	""" implements Nash-Sutcliffe Model Efficiency Coefficient where x is modeled and y is observed"""
	return 1 - np.sum((x - y)**2) / np.sum((y - y.mean())**2)

@metric
def normalized_nash_sutcliffe_efficiency(x, y):
	""" implements normalized nash_sutcliffe_efficiency """
	return 1.0 / (2 - nash_sutcliffe_efficiency(x, y))

@metric
def kling_gupta_efficiency(x, y, sr=1, sa=1, sb=1):
	""" implements kling gupta Efficiency where x is modeled and y = observed """
	alpha = x.std() / y.std()
	beta = x.mean() / y.mean()
	r, p = stats.pearsonr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return 1 - np.sqrt( (sr * (r - 1.0))**2 + (sa * (alpha - 1.0))**2 + (sb * (beta - 1.0))**2 )

@metric
def kling_gupta_components(x, y, sr=1, sa=1, sb=1, component='all' ):
	""" implements kling gupta Efficiency where x is modeled and y = observed """
	assert component in ['r', 'a', 'b', 'all'], 'invalid component {}'.format(component)
	if component == 'a':
		return x.std() / y.std()
	if component == 'b':
		return x.mean() / y.mean()
	if component == 'r':
		return stats.pearsonr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))[0]
	return np.asarray([x.std() / y.std(), x.mean() / y.mean(), stats.pearsonr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))[0]])

@metric
def spearman(x, y):
	"""abstracts out coefficient from stats.spearmanr"""
	coef, p = stats.spearmanr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return coef

@metric
def spearman_p(x, y):
	"""abstracts out p from stats.spearmanr"""
	coef, p = stats.spearmanr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return p

@metric
def pearson(x, y):
	"""abstracts out coefficient from stats.pearsonr"""
	coef, p = stats.pearsonr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return coef

@metric
def pearson_p(x, y):
	"""abstracts out p from stats.pearsonr"""
	coef, p = stats.pearsonr(np.squeeze(x).astype(float), np.squeeze(y).astype(float))
	return p
