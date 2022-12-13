from .emp_dist import EmpiricalDistribution

def CRPSS(predicted_cdfs, observations, quantiles):
    ecdf = EmpiricalDistribution().fit(observations)
    heaviside = ecdf.transform_to_heaviside(observations, quantiles)
    xs = ecdf.invcdf(quantiles)
    clim = ecdf.cdf(xs)
    clim_crps =  (clim - heaviside)**2 
    clim_crps = clim_crps.sum(axis=-1).mean()
    pred_crps = (predicted_cdfs - heaviside)**2
    pred_crps = pred_crps.sum(axis=-1).mean() 
    pred_crpss = 1 - pred_crps / clim_crps
    return clim_crps, pred_crps, pred_crpss
