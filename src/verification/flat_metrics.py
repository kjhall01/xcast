import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss, confusion_matrix
from .base_verification import metric
from scipy.special import ndtr
import itertools


def ordinal_weighted_logarithm_score(predicted, observed, eps=np.finfo('float').eps):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    correct_categories = np.where(observed == 1)

    observed = np.abs(observed - eps )
    predicted = np.abs(predicted - eps)

    ccs = np.argmax(observed, axis=-1)
    clim_base_rate = np.nanmean(observed, axis=0).reshape(1, -1)
    ls = 1 - np.log(predicted) / np.log(clim_base_rate)
    lss = ls[correct_categories].reshape(-1,1)
    ndxs = np.ones(observed.shape, dtype=float) * np.arange(observed.shape[-1]).reshape(1, -1)
    dists = np.abs(ndxs - ccs.reshape(-1,1))
    penalties = np.nanmean(dists * ls, axis=-1).reshape(-1,1)
    return np.nanmean(lss - penalties)

def logarithm_skill_score(predicted, observed, eps=np.finfo('float').eps):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    correct_categories = np.where(observed == 1)
    observed = np.abs(observed - eps )
    predicted = np.abs(predicted - eps)
    ccs = np.argmax(observed, axis=-1)
    clim_base_rate = np.nanmean(observed, axis=0).reshape(1, -1)
    ls = 1 - np.log(predicted) / np.log(clim_base_rate)
    lss = ls[correct_categories].reshape(-1,1)
    return np.nanmean(lss)


def generalized_receiver_operating_characteristics_score(predicted, observed):
    samples, classes = predicted.shape
    observed_cats = np.squeeze(np.argmax(observed, axis=1))
    cat_comps = np.array(np.meshgrid(
        observed_cats, observed_cats)).T.reshape(-1, 2)
    ndxs = np.array(np.meshgrid(np.arange(samples),
                    np.arange(samples))).T.reshape(-1, 2)
    pairs = ndxs[cat_comps[:, 0] < cat_comps[:, 1]]
    predictions1 = predicted[pairs[:, 0], :]
    predictions2 = predicted[pairs[:, 1], :]
    denominators = np.ones((predictions1.shape[0], 1)) - np.sum(
        predictions1 * predictions2, axis=-1).reshape(-1, 1)
    numerators = np.zeros((predictions1.shape[0], 1))
    for i in range(classes-1):
        numerators += np.sum(predictions1[:, i].reshape(-1, 1)
                             * predictions2[:, (i+1):], axis=-1).reshape(-1, 1)
    hit_scores = numerators / denominators
    hit_scores[hit_scores < 0.5] = 0
    hit_scores[hit_scores > 0.5] = 1
    hit_scores[hit_scores == 0.5] = 0.5
    if pairs.shape[0] == 0:
        return np.asarray([np.nan])
    return np.sum(hit_scores) / pairs.shape[0]



def log_likelihood(predicted, observed):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    n, k = predicted.shape
    residuals = np.squeeze(predicted - observed)
    return np.squeeze(-(n * 1/2) * (1 + np.log(2 * np.pi)) - (n / 2) * np.log(residuals.dot(residuals) / n))

def bias_ratio(predicted, observed):
    return np.nanmean(predicted) / np.nanmean(observed)

def normalized_centered_root_mean_squared_error(predicted, observed):
    denom = np.nanmean(observed)
    if denom == 0:
        return np.nan
    num = np.nanmean( (predicted - observed - np.nanmean(predicted - observed) )**2 )
    return np.sqrt(num) / denom

def akaike_information_criterion(predicted, observed):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    ll = log_likelihood(predicted, observed)
    n, k = predicted.shape
    return np.squeeze((-2 * ll) + (2 * k))


def brier_score(predicted, observed):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan for i in range(predicted.shape[1])])
    return np.asarray([brier_score_loss(observed[:, i].reshape(-1, 1), predicted[:, i].reshape(-1, 1)) for i in range(predicted.shape[1])])


def rank_probability_score(predicted, observed):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    return np.squeeze(np.mean(np.sum((observed-predicted)**2, axis=-1), axis=0))

def ignorance(predicted, observed):
    """ where predicted is predicted and observed is observed """
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    observed = np.argmax(observed, axis=1)
    logs = np.log(predicted)
    ign = 0
    for i in range(predicted.shape[0]):
        ign += logs[i, observed[i]]
    return np.squeeze(-1 * ign / float(predicted.shape[0]))


def hansen_kuiper(predicted, observed):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan for i in range(predicted.shape[1])])
    cm = confusion_matrix(np.squeeze(np.argmax(observed, axis=1)), np.squeeze(
        np.argmax(predicted, axis=1)), labels=[i for i in range(observed.shape[1])])
    ret = []
    for i in range(observed.shape[1]):
        try:
            total = np.sum(cm[i, :])
            hits = cm[i, i]
            misses = total - hits
            negs = np.delete(cm, i, axis=0)
            false_alarms = np.sum(negs[:, i])
            correct_negatives = np.sum(negs) - false_alarms
            hansen_kuiper_score = (float(hits) / float(hits + misses)) - \
                (float(false_alarms) / float(false_alarms + correct_negatives))
            ret.append(hansen_kuiper_score)
        except:
            ret.append(np.nan)
    return np.squeeze(np.asarray(ret))


def heidke_skill_score(predicted, observed):
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan for i in range(predicted.shape[1])])
    cm = confusion_matrix(np.squeeze(np.argmax(observed, axis=1)), np.squeeze(
        np.argmax(predicted, axis=1)), labels=[i for i in range(observed.shape[1])])
    ret = []
    for i in range(observed.shape[1]):
        try:
            n, k = predicted.shape
            total = np.sum(cm[i, :])
            hits = cm[i, i]
            misses = total - hits
            negs = np.delete(cm, i, axis=0)
            false_alarms = np.sum(negs[:, i])
            correct_negatives = np.sum(negs) - false_alarms
            expected_correct = (float(hits+ misses)*float(hits+ false_alarms))+ \
                                          (float(correct_negatives+ misses)*float(correct_negatives+ false_alarms))/n
            heidke_skill_score = (float(hits + correct_negatives) - float(expected_correct))/(float(n-expected_correct)) 
            ret.append(heidke_skill_score)
        except:
            ret.append(np.nan)
    return np.squeeze(np.asarray(ret))


def kendalls_tau(predicted, observed):
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
        rs.append(stats.pointbiserialr(np.squeeze(
            observed[i]), np.squeeze(predicted[i])))
    return np.asarray(rs)


def index_of_agreement(predicted, observed):
    """implements index of agreement metric"""
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    return 1 - np.sum((observed-predicted)**2) / np.sum((np.abs(observed - np.nanmean(predicted)) + np.abs(predicted - np.nanmean(predicted)))**2)


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
    alpha = predicted.var() / observed.var()
    beta = predicted.mean() / observed.mean()
    r, p = stats.pearsonr(np.squeeze(predicted).astype(
        float), np.squeeze(observed).astype(float))
    return 1 - np.sqrt((sr * (r - 1.0))**2 + (sa * (alpha - 1.0))**2 + (sb * (beta - 1.0))**2)


def kling_gupta_components(predicted, observed, sr=1, sa=1, sb=1, component='all'):
    """ implements kling gupta Efficiencobserved where predicted is modeled and observed = observed """
    if np.isnan(np.min(predicted)) or np.isnan(np.min(observed)):
        return np.asarray([np.nan])
    assert component in ['r', 'a', 'b',
                         'all'], 'invalid component {}'.format(component)
    if component == 'a':
        return predicted.var() / observed.var()
    if component == 'b':
        return predicted.mean() / observed.mean()
    if component == 'r':
        return stats.pearsonr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))[0]
    return np.asarray([predicted.var() / observed.var(), predicted.mean() / observed.mean(), stats.pearsonr(np.squeeze(predicted).astype(float), np.squeeze(observed).astype(float))[0]])
