from .einstein_epoelm import EPOELM 
import numpy as np 
from sklearn.model_selection import KFold 
import datetime as dt 
from .crpss import CRPSS

HYPERPARAMETER_TUNING = {
    'c': [-10, -5,  -3, 3, 5, 10],
    'hidden_layer_size': [3, 5, 10, 15, 25, 50, 100, 250],
    'activation': ['relu', 'sigm', 'tanh', 'lin'],
    'preprocessing': ['minmax', 'std', 'none'],
    'encoding': [ 'binary', 'nonexceedance'],
    'quantiles': [  [0.2, 0.4, 0.6, 0.8], [0.33, 0.67], [0.01, 0.5, 0.99], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
}

def tune(x, y): 
    best_crpss = -9999
    total = len(HYPERPARAMETER_TUNING['c'])  * len(HYPERPARAMETER_TUNING['hidden_layer_size']) * len(HYPERPARAMETER_TUNING['activation']) * len(HYPERPARAMETER_TUNING['preprocessing']) * len(HYPERPARAMETER_TUNING['encoding']) * len(HYPERPARAMETER_TUNING['quantiles'])
    i = 1
    for c in HYPERPARAMETER_TUNING['c']:
        for hidden_layer_size in HYPERPARAMETER_TUNING['hidden_layer_size']:
            for activation in HYPERPARAMETER_TUNING['activation']:
                for preprocessing in HYPERPARAMETER_TUNING['preprocessing']:
                    for encoding in HYPERPARAMETER_TUNING['encoding']:
                        for quantiles in HYPERPARAMETER_TUNING['quantiles']:
                            params = {
                                'c': c, 
                                'hidden_layer_size': hidden_layer_size,
                                'activation': activation, 
                                'preprocessing': preprocessing, 
                                'encoding': encoding, 
                                'quantiles': quantiles
                            }
                            score = get_score(params, x, y)
                            print('Progress: {}%'.format(round( i / float(total) * 100), 2), end='\r')
                            i += 1
                            if True:
                                print("HyperParameters:  {} ".format(params))
                                print("CRPSS: {}".format(score))
                                print()
                            if score > best_crpss: 
                                best_crpss = score
                                best_params = params
    print()
    return best_params, best_crpss

def get_score(params, x, y):
    quantiles = np.linspace(0.01, 0.99, 99)
    indices = np.arange(x.shape[0])
    ndx = 1
    xval_y, xval_predicted_cdfs = [], []
    thresholds = np.nanquantile(y, quantiles, method='midpoint')
    kf = KFold(n_splits=5)
    for xtrainndx, xtestndx in kf.split(indices):

        xtrain, ytrain = x[xtrainndx, :], y[xtrainndx, :]
        xtest, ytest = x[xtestndx, :], y[xtestndx, :]
        xval_y.append(ytest)

        epoelm = EPOELM(**params)
        start = dt.datetime.now()
        epoelm.fit(xtrain, ytrain)
        predicted_cdfs = epoelm.predict_proba(xtest, threshold=thresholds)
        end = dt.datetime.now()

        xval_predicted_cdfs.append(predicted_cdfs)
        ndx += 1

    xval_predicted_cdfs = np.vstack(xval_predicted_cdfs)
    xval_y = np.vstack(xval_y)
    clim_crps, pred_crps, crpss = CRPSS(xval_predicted_cdfs, xval_y, quantiles)
    return crpss