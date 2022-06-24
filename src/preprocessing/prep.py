from numpy import vectorize
import xarray as xr
from scipy.stats import norm, gamma, skew
from scipy.interpolate import interp1d
import statsmodels.distributions.empirical_distribution as edf
import numpy as np
from ..core.utilities import guess_coords, check_all
import sys


def svd_flip(u, v):
    max_abs_rows = np.argmax(np.abs(v), axis=1)
    signs = np.sign(v[range(v.shape[0]), max_abs_rows])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v


def svd_flip_v(v):
    signs = []
    for i in range(v.shape[0]):
        if np.abs(np.max(v[i, :])) < np.abs(np.min(v[i, :])):
            v[i, :] *= -1
            signs.append(-1)
        else:
            signs.append(1)
    return v, np.asarray(signs)


def invcdf(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, dist=norm):
    x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    def _xr_tppf(x):
        return dist.ppf(x)  # , loc=y.mean(), scale=y.std())
    return xr.apply_ufunc(_xr_tppf, X, input_core_dims=[[x_sample_dim]], output_core_dims=[[x_sample_dim]], keep_attrs=True, vectorize=True)


def cdf(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, dist=norm):
    x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    def _xr_cdf(x):
        return dist.cdf(x)  # , loc=y.mean(), scale=y.std())
    return xr.apply_ufunc(_xr_cdf, X, input_core_dims=[[x_sample_dim]], output_core_dims=[[x_sample_dim]], keep_attrs=True, vectorize=True)


def percentile(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    X1 = X.rank(x_sample_dim, pct=True)
    X1 = X1.where(X1 < 1, other=0.9999)
    X1 = X1.where(X1 > 0, other=0.0001)
    return X1


def transform(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, dist=norm):
    x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    pctl = percentile(X, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim,
                      x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
    return invcdf(pctl.dropna(x_sample_dim), x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim, dist=dist)


def invcdf(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, dist=norm):
    x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    def _xr_tppf(x):
        return dist.ppf(x)  # , loc=y.mean(), scale=y.std())
    return xr.apply_ufunc(_xr_tppf, X, input_core_dims=[[x_sample_dim]], output_core_dims=[[x_sample_dim]], keep_attrs=True, vectorize=True)


class GammaTransformer:
    def __init__(self, destination=norm):
        self.destination_distribution = destination

    def fit(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        self._fit_source(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    def _fit_source(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        """Makes an 'empirical cdf' function at each point in space"""
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        def _make_dist(x):
            # censor x where x < eps
            x = x[x > np.finfo(float).eps]
            #  where epsilon = machine precision, and tolerance = sqrt(epsilon)
            # check that skewness  4*( E[logX] - log(E[X])  ) > 0 for MLE fitting
            a = 4 * (np.log(x.mean()) - np.mean(np.log(x)))
            # here we use Maximum Likelihood Estimate
            if a > np.sqrt(np.finfo(float).eps):
                alpha = (1 + np.sqrt(1 + (a/3)))/a  #
                beta = np.mean(x) / alpha
                loc = np.min(x)
                method = 'mle'
            else:  # and here, Method of Moments
                # print(x)
                method = 'mm'
                if np.var(x) > np.finfo(float).eps:
                   # print('var > eps')
                    beta = np.var(x) / np.mean(x)
                    alpha = np.mean(x) / beta
                    loc = np.min(x)
                else:
                    alpha, beta, loc = 0, 0.0001, 0
            return gamma(alpha, scale=beta)

        self.dists = xr.apply_ufunc(_make_dist, X, input_core_dims=[
                                    [x_sample_dim]], output_core_dims=[[]], keep_attrs=True, vectorize=True)

    def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        def _apply_cdf(x, dist):
            return dist.cdf(x)

        percentiles = xr.apply_ufunc(_apply_cdf, X, self.dists, input_core_dims=[
                                     [x_sample_dim], []], output_core_dims=[[x_sample_dim]], keep_attrs=True, vectorize=True)
        return invcdf(percentiles, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim, dist=self.destination_distribution)

    def inverse_transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        percentiles = cdf(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None,
                          x_feature_dim=None, dist=self.destination_distribution)

        def _xr_invert(x, dist):
            return dist.ppf(x)
        return xr.apply_ufunc(_xr_invert, percentiles, self.dists, input_core_dims=[[x_sample_dim], []], output_core_dims=[[x_sample_dim]], keep_attrs=True, vectorize=True)


class EmpiricalTransformer:
    def __init__(self, destination=norm):
        self.destination_distribution = destination

    def fit(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        self._make_cdfs(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        self._make_edf_invcdfs(X, x_lat_dim, x_lon_dim,
                               x_sample_dim,  x_feature_dim)

    def _make_cdfs(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        """Makes an 'empirical cdf' function at each point in space"""
        def _make_invcdf(x):
            empirical_cdf = edf.ECDF(np.squeeze(x))
            return empirical_cdf
        self.cdfs = xr.apply_ufunc(_make_invcdf, X, input_core_dims=[
                                   [x_sample_dim]], output_core_dims=[[]], keep_attrs=True, vectorize=True)

    def _make_edf_invcdfs(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        """Makes an 'empirical invcdf' function at each point in space"""
        def _make_invcdf(x):
            empirical_cdf = edf.ECDF(np.squeeze(x))
            slope_changes = sorted(set(np.squeeze(x)))
            cdf_vals_at_slope_changes = [
                empirical_cdf(i) for i in slope_changes]
            return interp1d(cdf_vals_at_slope_changes, slope_changes, fill_value='extrapolate')
        self.invcdfs = xr.apply_ufunc(_make_invcdf, X, input_core_dims=[
                                      ['T']], output_core_dims=[[]], keep_attrs=True, vectorize=True)

    def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        def _xr_invert1(x, cdf1):
            return cdf1(x)
        percentiles = xr.apply_ufunc(_xr_invert1, X, self.cdfs, input_core_dims=[[x_sample_dim], [
                                     x_sample_dim]], output_core_dims=[[x_sample_dim]], keep_attrs=True, vectorize=True)  # percentile(X )
        return invcdf(percentiles, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim, dist=self.destination_distribution)

    def inverse_transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        percentiles = cdf(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None,
                          x_feature_dim=None, dist=self.destination_distribution)

        def _xr_invert(x, invcdf1):
            return invcdf1(x)
        return xr.apply_ufunc(_xr_invert, percentiles, self.invcdfs, input_core_dims=[[x_sample_dim], []], output_core_dims=[[x_sample_dim]], keep_attrs=True, vectorize=True)
