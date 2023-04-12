import xarray as xr
from ..core.utilities import *
import numpy as np

import pandas as pd 

class LeaveOneYearOut:
    def __init__(self, X, Y, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None ):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim )
        y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim )
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        self.samples = X.shape[list(X.dims).index(x_sample_dim)]
        self.x_sample_dim, self.y_sample_dim = x_sample_dim, y_sample_dim
        assert X.shape[list(X.dims).index(x_sample_dim)] == Y.shape[list(Y.dims).index(y_sample_dim)], 'X and Y must have the same number of samples - they do not'
        self.X, self.Y = X, Y
        self.years  = list(set([ pd.Timestamp(i).year for i in self.X.coords[x_sample_dim].values ]))
        self.index = 0

    def get_next(self):
        if self.index >= len(self.years): 
            raise StopIteration
        xtest_dct = {self.x_sample_dim: slice(pd.Timestamp(self.years[self.index], 1, 1), pd.Timestamp(self.years[self.index], 12, 31))}
        ytest_dct = {self.y_sample_dim: slice(pd.Timestamp(self.years[self.index], 1, 1), pd.Timestamp(self.years[self.index], 12, 31))}
        xtrain_dct1 = {self.x_sample_dim: slice(None, pd.Timestamp(self.years[self.index], 1, 1) ) }
        xtrain_dct2 = {self.x_sample_dim: slice(pd.Timestamp(self.years[self.index], 12, 31), None)}
        ytrain_dct1 = {self.y_sample_dim: slice(None, pd.Timestamp(self.years[self.index], 1, 1) ) }
        ytrain_dct2 = {self.y_sample_dim: slice(pd.Timestamp(self.years[self.index], 12, 31), None)}
        xtest = self.X.sel(**xtest_dct)
        ytest = self.Y.sel(**ytest_dct)
        xtrain1 = self.X.sel(**xtrain_dct1)
        xtrain2 = self.X.sel(**xtrain_dct2)
        ytrain1 = self.Y.sel(**ytrain_dct1)
        ytrain2 = self.Y.sel(**ytrain_dct2)
        self.index += 1
        return  xr.concat([xtrain1, xtrain2], self.x_sample_dim), xr.concat([ytrain1, ytrain2], self.y_sample_dim), xtest,  ytest

       

    def __iter__(self):
        return self 
    
    def __next__(self): 
        return self.get_next()
