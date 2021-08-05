import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from .base_verification import *

#probabilistic scores
def roc_auc_score_(x, y):
	try:
		ret =  roc_auc_score(x, y, average=None, multi_class='ovr', labels=[0,1,2])
		return ret
	except:
		return np.asarray([0,0,0])

def f1_score_(x, y):
	try:
		ret =  f1_score(np.argmax(y, axis=-1), np.argmax(x, axis=-1), average=None, labels=[0,1,2])
		return ret
	except:
		return np.asarray([0,0,0])

def average_precision_score_(x, y):
	return average_precision_score(x, y, average=None)

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




class MeanAbsoluteError(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = mean_absolute_error

class MeanSquaredError(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = mean_squared_error

class PearsonSignificance(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = pearson_p_

class PearsonCoefficient(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = pearson_coef_

class SpearmanSignificance(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = spearman_p_

class SpearmanCoefficient(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = spearman_coef_

class IndexOfAgreement(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = index_of_agreement_

class AveragePrecision(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = average_precision_score_

class AUROC(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = roc_auc_score_

class F1(BaseMetric):
	def __init__(self, use_dask=False):
		super().__init__(use_dask=use_dask)
		self.function = f1_score_
