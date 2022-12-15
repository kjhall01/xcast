from .emp_dist import EmpiricalDistribution
import numpy as np
import matplotlib.pyplot as plt

def crps(x, standardize_y=False):
    if standardize_y:
        x = (x - x.mean()) / x.std()
    x = np.squeeze(x)
    indicator_functions = np.vstack([i <= np.sort(x) for i in x]).astype(float)
    deltas = np.diff(np.sort(x))
    ecdf = np.asarray([ (x < i).mean() for i in np.sort(x) ] )
    squared_diffs = (indicator_functions - ecdf)**2
    rectangles = deltas * squared_diffs[:, :-1]
    crpses = np.nansum(rectangles, axis=-1)
    return np.nanmean(crpses)

def CRPSS(predicted_cdfs, observations, quantiles, standardize_y=False):
    if standardize_y:
        observations = (observations - observations.mean()) / observations.std()
    ecdf = EmpiricalDistribution().fit(observations)
    heaviside = ecdf.transform_to_heaviside(observations, quantiles)
    xs = ecdf.invcdf(quantiles)
    clim = ecdf.cdf(xs)


    clim_crps =  (clim - heaviside)**2
    clim_crps = clim_crps[:, :-1] * np.diff(xs)
    #plt.plot(xs, heaviside[0,:], label='indicator')
    #plt.plot(xs,clim, label='ecdf')
    #plt.title('crps: {}'.format(clim_crps[0,:].sum()))
    #plt.legend()
    #plt.show()
    clim_crps = clim_crps.sum(axis=-1).mean()
    pred_crps = (predicted_cdfs - heaviside)**2
    pred_crps = pred_crps[:, :-1] * np.diff(xs)
    pred_crps = pred_crps.sum(axis=-1).mean()
    pred_crpss = 1 - pred_crps / clim_crps
    return clim_crps, pred_crps, pred_crpss
