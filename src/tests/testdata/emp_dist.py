import statsmodels.distributions.empirical_distribution as edf
from scipy.interpolate import interp1d
import numpy as np


class EmpiricalDistribution:
    def __init__(self):
        pass

    def fit(self, x):
        self.cdf = edf.ECDF(np.squeeze(x))
        self.invcdf = self._make_invcdf(x)
        self.min, self.max = x.min(), x.max()
        return self

    def _make_invcdf(self, x):
        slope_changes = sorted(set(np.squeeze(x)))
        cdf_vals_at_slope_changes = [self.cdf(i) for i in slope_changes]
        return interp1d(cdf_vals_at_slope_changes, slope_changes, fill_value='extrapolate')

    def transform_to_heaviside(self, x, quantiles):
        ret = np.ones((x.shape[0], quantiles.shape[0]))
        xs = self.invcdf(quantiles)
        retx = ret*x
        retxs = ret * xs
        ret[ retx > retxs] = 0
        return ret
