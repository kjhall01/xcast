from .einstein_epoelm import EPOELM
import numpy as np
from sklearn.model_selection import KFold
import datetime as dt
from .crpss import CRPSS, crps
import itertools

HYPERPARAMETER_TUNING = {
    'c': [-10, -5,  -3, 3, 5, 10],
    'hidden_layer_size': [3, 5, 10, 15, 25, 50, 100, 250],
    'activation': ['relu', 'sigm', 'tanh', 'lin'],
    'preprocessing': ['minmax', 'std', 'none'],
    'encoding': [ 'binary', 'nonexceedance'],
    'quantiles': [  [0.2, 0.4, 0.6, 0.8], [0.33, 0.67], [0.01, 0.5, 0.99], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
}

def get_crpss(params, x, y, estimator=EPOELM):
    quantiles = np.linspace(0.01, 0.99, 99)
    indices = np.arange(x.shape[0])
    ndx = 1
    xval_y, xval_predicted_cdfs = [], []
    pred_crps = []

    #thresholds = np.nanquantile(y, quantiles, method='midpoint')
    kf = KFold(n_splits=15)
    for xtrainndx, xtestndx in kf.split(indices):
        xtrain, ytrain = x[xtrainndx, :], y[xtrainndx, :]
        xtest, ytest = x[xtestndx, :], y[xtestndx, :]
        xval_y.append(ytest)
        epoelm = estimator(**params)
        epoelm.fit(xtrain, ytrain)
        pred_crps.append(epoelm.crps(xtest, ytest))

        predicted_cdfs = epoelm.predict_proba(xtest, quantile=quantiles)
        xval_predicted_cdfs.append(predicted_cdfs)
        ndx += 1

    xval_predicted_cdfs = np.vstack(xval_predicted_cdfs)
    xval_y = np.vstack(xval_y)
    clim_crps, pred_crps1, crpss = CRPSS(xval_predicted_cdfs, xval_y, quantiles, standardize_y=epoelm.standardize_y)
    pred_crps = np.vstack(pred_crps).mean()
    return 1 - pred_crps / crps(y, standardize_y=epoelm.standardize_y), 1 - pred_crps1 / clim_crps

def get_rmse(params, x, y, estimator=EPOELM):
    indices = np.arange(x.shape[0])
    ndx = 1
    xval_y, xval_predicted_cdfs = [], []
    kf = KFold(n_splits=5)
    for xtrainndx, xtestndx in kf.split(indices):
        xtrain, ytrain = x[xtrainndx, :], y[xtrainndx, :]
        xtest, ytest = x[xtestndx, :], y[xtestndx, :]
        xval_y.append(ytest)
        epoelm = estimator(**params)
        epoelm.fit(xtrain, ytrain)
        predicted = epoelm.predict(xtest)
        xval_predicted_cdfs.append(predicted)
        ndx += 1

    xval_predicted_cdfs = np.vstack(xval_predicted_cdfs)
    xval_y = np.vstack(xval_y)
    return  -1*np.sqrt( np.nanmean(  (xval_predicted_cdfs - xval_y )**2) )

def tune(x, y, params=HYPERPARAMETER_TUNING, scorer=get_crpss, verbosity=1):
    best_crpss = -9999
    i = 1
    keys = sorted([key for key in params.keys()])
    lists = [ params[key] for key in keys]
    combos = [ combo for combo in itertools.product(*lists) ]
    total = len(combos)
    for combo in combos:
        params = {keyword: combo[param] for param, keyword in enumerate(keys) }
        score, old_score = scorer(params, x, y)
        if verbosity > 1:
            print("\nHyperParameters:  {} ".format(params))
            print("CRPSS: {}".format(score))
        if score > best_crpss:
            best_crpss = score
            best_old_score = old_score
            best_params = params
        if verbosity > 0:
            print('Progress: {}% (Best Score: {}; Associated Old-Style: {})'.format(round( i / float(total) * 100, 2), round(best_crpss, 3), round(best_old_score, 3)), end='\r')
            i += 1
    if verbosity == 1:
        print()
    return best_params, best_crpss, best_old_score
