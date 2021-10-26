import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, confusion_matrix
from .base_verification import *
from .flat_metrics import *

@metric
def BrierScoreLoss(predicted, observed):
	return brier_score_loss(predicted, observed)

@metric
def BrierScore(predicted, observed):
	return brier_score(predicted, observed)

@metric
def RankProbabilityScore(predicted, observed):
	return rank_probability_score(predicted, observed)

@metric
def ContinuousRankProbabilityScore(predicted, observed):
	return continuous_rank_probability_score(predicted, observed)

@metric
def Ignorance(predicted, observed):
	return ignorance(predicted, observed)

@metric
def PointBiserialCorrelation(predicted, observed):
	return point_biserial_correlation(predicted, observed)

@metric
def HansenKuiper(predicted, observed):
	return hansen_kuiper(predicted, observed)

@metric
def MeanAbsolutePercentError(predicted, observed):
	return mean_absolute_percentage_error(observed, predicted)

@metric
def KendallsTau(predicted,observed):
	return kendalls_tau(predicted, observed )

@metric
def BayesianInformationCriterion(predicted, observed):
	return np.asarray( [ bayesian_information_criterion(predicted[:,i].reshape(-1,1), observed[:,i].reshape(-1,1) ) for i in range(predicted.shape[1]) ] )

@metric
def AkaikeInformationCriterion(predicted, observed):
	return np.asarray([ akaike_information_criterion(predicted[:,i].reshape(-1,1), observed[:,i].reshape(-1,1)) for i in range(predicted.shape[1]) ] )

@metric
def LogLikelihood(predicted, observed):
	return np.asarray([ log_likelihood(predicted[:,i].reshape(-1,1), observed[:,i].reshape(-1,1) ) for i in range(predicted.shape[1]) ])

@metric
def RocAuc(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	ret =  roc_auc_score(predicted, observed, average=None, multi_class='ovr', labels=[0,1,2])
	return ret


@metric
def GeneralizedROC(predicted, observed):
	return generalized_roc(predicted, observed)

@metric
def F1(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	ret =  f1_score(np.argmax(observed, axis=-1), np.argmax(predicted, axis=-1), average=None, labels=[0,1,2])
	return ret


@metric
def AveragePrecision(predicted, observed):
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	return average_precision_score(predicted, observed, average=None)
@metric
def IndexOfAgreement(predicted, observed):
	"""implements index of agreement metric"""
	return index_of_agreement(predicted, observed )

@metric
def NashSutcliffeEfficiency(predicted, observed):
	""" implements Nash-Sutcliffe Model Efficiencobserved Coefficient where predicted is modeled and observed is observed"""
	return nash_sutcliffe_efficiency(predicted, observed )

@metric
def NormalizedNashSutcliffeEfficiency(predicted, observed):
	""" implements normalized nash_sutcliffe_efficiencobserved """
	return normalized_nash_sutcliffe_efficiency(predicted, observed )

@metric
def KlingGuptaEfficiency(predicted, observed, sr=1, sa=1, sb=1):
	""" implements kling gupta Efficiencobserved where predicted is modeled and observed = observed """
	return kling_gupta_efficiency(predicted, observed, sr=sr, sa=sa, sb=sb)

@metric
def KlingGuptaComponents(predicted, observed, sr=1, sa=1, sb=1, component='all' ):
	""" implements kling gupta Efficiencobserved where predicted is modeled and observed = observed """
	return kling_gupta_components(predicted, observed, sr=sr, sa=sa, sb=sb, component=component)

@metric
def Spearman(predicted, observed):
	"""abstracts out coefficient from stats.spearmanr"""
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	coef, p = stats.spearmanr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))
	return coef

@metric
def SpearmanP(predicted, observed):
	"""abstracts out p from stats.spearmanr"""
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	coef, p = stats.spearmanr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))
	return p

@metric
def Pearson(predicted, observed):
	"""abstracts out coefficient from stats.pearsonr"""
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	coef, p = stats.pearsonr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))
	return coef

@metric
def PearsonP(predicted, observed):
	"""abstracts out p from stats.pearsonr"""
	if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
		return np.asarray([np.nan])
	coef, p = stats.pearsonr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))
	return p

generalized_probabilistic_metrics = [ RankProbabilityScore, ContinuousRankProbabilityScore, Ignorance, GeneralizedROC, F1, AveragePrecision,  ]
categorical_probabilistic_metrics = [RocAuc, BrierScore, PointBiserialCorrelation, HansenKuiper, PointBiserialCorrelation]

single_output_deterministic_metrics = [PearsonP, Pearson, SpearmanP, Spearman, KlingGuptaEfficiency, NormalizedNashSutcliffeEfficiency, NashSutcliffeEfficiency, IndexOfAgreement, BayesianInformationCriterion, AkaikeInformationCriterion, LogLikelihood]
multiple_output_deterministic_metrics = [KlingGuptaComponents, BayesianInformationCriterion, AkaikeInformationCriterion, LogLikelihood]
