import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, confusion_matrix
from .base_verification import *
from scipy.special import ndtr
import itertools

def generalized_roc(predicted, observed):
	samples, classes = predicted.shape
	observed = np.argmax(observed, axis=1)
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	pairs = []
	for i, j in itertools.permutations(range(samples), 2):
		if observed[i] > observed[j]:
			pairs.append([j, i])
	pairs = np.asarray(pairs)
	if len(pairs) == 0:
		return np.asarray([1])
	predictions1 = predicted[pairs[:,0], :]
	predictions2 = predicted[pairs[:,1], :]
	numerators = np.zeros((predictions1.shape[0], 1))

	for i in range(classes-1):
		for j in range(i+1, classes):
			pr, ps = predictions1[:,i].reshape(-1, 1), predictions2[:, j].reshape(-1, 1)
			numerators += pr*ps

	denominators = np.ones((predictions1.shape[0], 1))
	for i in range(classes):
		denominators -= predictions1[:,i].reshape(-1,1) * predictions2[:,i].reshape(-1,1)

	hit_scores = numerators.astype(float) / denominators.astype(float)
	hit_scores[hit_scores < 0.5] = 0
	hit_scores[hit_scores > 0.5] = 1
	hit_scores[hit_scores == 0.5] = 0.5
	return np.sum(hit_scores) / float(pairs.shape[0])

def log_likelihood(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	n, k =  predicted.shape
	residuals = np.squeeze(predicted - observed)
	return -(n * 1/2) * (1 + np.log(2 * np.pi)) - (n / 2) * np.log(residuals.dot(residuals) / n)

def akaike_information_criterion(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	ll = log_likelihood(predicted, observed)
	n, k = predicted.shape
	return (-2 * ll) + (2 * k)

def brier_score(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	return np.asarray([brier_score_loss(predicted[:,i], observed[:,i]) for i in predicted.shape[1] ])

def rank_probability_score(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	return np.mean(np.sum((observed-predicted)**2, axis=-1), axis=0)

def continuous_rank_probability_score(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	observed = np.cumsum(observed, axis=1)
	predicted = np.cumsum(predicted, axis=1)
	return np.nanmean(np.sum((predicted - observed)**2, axis=1), axis=0)

def ignorance(predicted, observed):
	""" where predicted is predicted and observed is observed """
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	observed = np.argmax(observed, axis=1)
	logs = np.log(predicted)
	ign = 0
	for i in range(predicted.shape[0]):
		ign += logs[i, observed[i]]
	return -1 * ign / float(predicted.shape[0])

def hansen_kuiper(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	cm = confusion_matrix(np.squeeze(np.argmax(observed, axis=1)), np.squeeze(np.argmax(predicted), axis=1), labels=[i for i in range(observed.shape[2])])
	ret = []
	for i in range(observed.shape[2]):
		total = np.sum(cm[i,:])
		hits = cm[i,i]
		misses = total - hits

		negs = np.delete(cm, i, axis=0)
		false_alarms = np.sum(negs[i,:])
		correct_negatives = np.sum(negs) - false_alarms
		hansen_kuiper_score = ( float(hits) / float(hits + misses) ) - (float(false_alarms) / float(false_alarms + correct_negatives))
		ret.append(hansen_kuiper_score)
	return np.asarray(ret)

def kendalls_tau(predicted,observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	predicted, observed = np.squeeze(predicted), np.squeeze(observed)
	return stats.kendalltau(predicted[predicted.argsort()][::-1].astype(float), observed[observed.argsort()][::-1].astype(float))


def bayesian_information_criterion(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	ll = log_likelihood(predicted, observed)
	n, k = predicted.shape
	return (-2 * ll) + (k * np.log(n))


def point_biserial_correlation(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	rs = []
	for i in range(predicted.shape[1]):
		rs.append(stats.pointbiserialr(np.squeeze(observed[i]), np.squeeze(predicted[i])))
	return np.asarray(rs)


def index_of_agreement(predicted, observed):
	"""implements index of agreement metric"""
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	return 1 - np.sum((observed-predicted)**2) / np.sum(  (np.abs(observed - np.nanmean(predicted)) + np.abs(predicted - np.nanmean(predicted)))**2)

def nash_sutcliffe_efficiency(predicted, observed):
	""" implements Nash-Sutcliffe Model Efficiencobserved Coefficient where predicted is modeled and observed is observed"""
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	return 1 - np.sum((predicted - observed)**2) / np.sum((observed - observed.mean())**2)

def normalized_nash_sutcliffe_efficiency(predicted, observed):
	""" implements normalized nash_sutcliffe_efficiencobserved """
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	return 1.0 / (2 - nash_sutcliffe_efficiency(predicted, observed))

def kling_gupta_efficiency(predicted, observed, sr=1, sa=1, sb=1):
	""" implements kling gupta Efficiencobserved where predicted is modeled and observed = observed """
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	alpha = predicted.std() / observed.std()
	beta = predicted.mean() / observed.mean()
	r, p = stats.pearsonr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))
	return 1 - np.sqrt( (sr * (r - 1.0))**2 + (sa * (alpha - 1.0))**2 + (sb * (beta - 1.0))**2 )

def kling_gupta_components(predicted, observed, sr=1, sa=1, sb=1, component='all' ):
	""" implements kling gupta Efficiencobserved where predicted is modeled and observed = observed """
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	assert component in ['r', 'a', 'b', 'all'], 'invalid component {}'.format(component)
	if component == 'a':
		return predicted.std() / observed.std()
	if component == 'b':
		return predicted.mean() / observed.mean()
	if component == 'r':
		return stats.pearsonr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))[0]
	return np.asarray([predicted.std() / observed.std(), predicted.mean() / observed.mean(), stats.pearsonr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))[0]])
