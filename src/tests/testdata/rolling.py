import pandas as pd
import xarray as xr
from ...core.utilities import guess_coords, check_all
from ...preprocessing import quantile

class RollingOneHotEncoder:
    def __init__(self, agg_period=14, buffer=14, low=1.0/3.0, high=2.0/3.0, base_year=1999, n_years_to_search=30):
        self.low_thresh, self.high_thresh = low, high
        self.base_year = base_year
        self.agg_period = agg_period
        self.buffer = buffer
        self.n_years_to_search=n_years_to_search

    def fit(self, X, quantile_method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        """ where X is ytrain_year"""
        # loops over each calendar year in X - fits a 6-week period rolling climatology
        self.low_thresholds = {}
        self.high_thresholds = {}

        # sample dim coords must be convertible to pd.Timestamp
        doys = sorted(list( set([ pd.Timestamp(ii).strftime('%m-%d') for ii in X.coords[x_sample_dim].values])  ))

        low_threshs = []
        high_threshs = []
        coord = []

        for doy in doys:
            data = []
            mm, dd = [int(iii) for iii in doy.split('-')]
            init = pd.Timestamp(self.base_year, mm, dd )
            coord.append(init)
            low = init - pd.Timedelta(days=self.buffer)
            high = init + pd.Timedelta(days=self.buffer + self.agg_period)

            for j in range(self.n_years_to_search):
                low2 = low + pd.Timedelta(days=365*(j+1))
                high2 = high + pd.Timedelta(days=365*(j+1))
                dcttime = {x_sample_dim: slice(low2, high2)}
                toap = X.sel(**dcttime)
                if toap.shape[list(toap.dims).index(x_sample_dim)] > 0:
                    data.append(toap)
            data = xr.concat(data, x_sample_dim)
            low_threshs.append( quantile(data, self.low_thresh, method=quantile_method,  x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim) )
            high_threshs.append( quantile(data, self.high_thresh, method=quantile_method,  x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim) )
        self.low_thresholds = xr.concat(low_threshs, x_sample_dim).assign_coords({x_sample_dim: coord})
        self.high_thresholds = xr.concat(high_threshs, x_sample_dim).assign_coords({x_sample_dim: coord})

    def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        ret = []
        dct = {x_sample_dim: 0, 'drop': True}
        form = xr.ones_like(X.isel(**dct))
        for init in X.coords[x_sample_dim].values:
            init = pd.Timestamp(init)
            init_dct = {x_sample_dim: init}
            bn = form.where( X.sel(**init_dct) <= self.low_thresholds.sel(**init_dct, method='nearest'), other=0)
            an = form.where( X.sel(**init_dct) > self.high_thresholds.sel(**init_dct, method='nearest'), other=0)
            nn = 1-an - bn
            ret.append( xr.concat([bn, nn, an], x_feature_dim).assign_coords({x_feature_dim: ['BN', 'NN', 'AN']}) )
        return xr.concat(ret, x_sample_dim).assign_coords({x_sample_dim: X.coords[x_sample_dim].values })




class RollingMinMax:
    def __init__(self, agg_period=14, buffer=14, base_year=1999, n_years_to_search=30):
        self.base_year = base_year
        self.agg_period = agg_period
        self.buffer = buffer
        self.n_years_to_search=n_years_to_search

    def fit(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        """ where X is ytrain_year"""

        # loops over each calendar year in X - fits a 6-week period rolling climatology
        self.low_thresholds = []
        self.high_thresholds = []
        doys = sorted(list( set([ pd.Timestamp(ii).strftime('%m-%d') for ii in X.coords[x_sample_dim].values])  ))
        coord = []

        for doy in doys:
            data = []
            mm, dd = [int(iii) for iii in doy.split('-')]
            init = pd.Timestamp(self.base_year, mm, dd )
            coord.append(init)

            low = init - pd.Timedelta(days=self.buffer)
            high = init + pd.Timedelta(days=self.buffer + self.agg_period)

            for j in range(self.n_years_to_search):
                low2 = low + pd.Timedelta(days=365*(j+1))
                high2 = high + pd.Timedelta(days=365*(j+1))
                dct = { x_sample_dim: slice( low2, high2 ) }
                toap = X.sel(**dct)
                if toap.shape[list(toap.dims).index(x_sample_dim)] > 0:
                    data.append(toap)
            data = xr.concat(data, x_sample_dim)
            self.low_thresholds.append( data.min(x_sample_dim) )
            self.high_thresholds.append(  data.max(x_sample_dim) )
        self.low_thresholds = xr.concat(self.low_thresholds, x_sample_dim).assign_coords({x_sample_dim: coord})
        self.high_thresholds = xr.concat(self.high_thresholds, x_sample_dim).assign_coords({x_sample_dim: coord})


    def transform(self, X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
        x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
            X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

        ret = []
        for init in X.coords[x_sample_dim].values:
            init = pd.Timestamp(init)
            dct = {x_sample_dim: init}
            lt = self.low_thresholds.sel(**dct, method='nearest')
            ht = self.high_thresholds.sel(**dct, method='nearest')
            scaled = ( X.sel(**dct) - lt ) / ( ht - lt ) * 2 -1
            ret.append( scaled )
        return xr.concat(ret, x_sample_dim).assign_coords({x_sample_dim: X.coords[x_sample_dim].values })
