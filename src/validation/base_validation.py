import xarray as xr
#import xskillscore as xs
from ..core.utilities import *
from ..core.progressbar import *
import numpy as np



class CrossValidator:
    def __init__(self, X, Y, window=1, step= 1,  x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, y_lat_dim=None, y_lon_dim=None, y_sample_dim=None, y_feature_dim=None ):
        x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim,  x_feature_dim )
        y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim,  y_feature_dim )
        check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
        check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
        self.samples = X.shape[list(X.dims).index(x_sample_dim)]
        self.x_sample_dim, self.y_sample_dim = x_sample_dim, y_sample_dim
        assert X.shape[list(X.dims).index(x_sample_dim)] == Y.shape[list(Y.dims).index(y_sample_dim)], 'X and Y must have the same number of samples - they do not'
        self.X, self.Y, self.window, self.index, self.step = X, Y, window, 0, step
        assert window % 2 == 1, 'Cross Validation Window must be odd'
        self.radius = window // 2 # count out from center to edge of window- ie 0 1 2 3 4 centered on two, this would be 2 

    def get_next(self):
        bottom_boundary, top_boundary = self.index - self.radius, self.index + self.radius + 1 # if this is 0, you get -2, 2 which includes -2, -1, 0, 1, 2 all in the sample 
        bottom_boundary = self.samples + bottom_boundary  if bottom_boundary < 0 else bottom_boundary # bottom boundary will never be greater than self.samples - 1 - self.radius 
        top_boundary = top_boundary - self.samples if top_boundary >= self.samples else top_boundary
        if self.index >= self.samples: 
            raise StopIteration
        if bottom_boundary > top_boundary: 
            x_bottom_of_window_high_indices = self.X.isel(**{self.x_sample_dim: slice(bottom_boundary, None)})
            x_top_of_window_low_indices = self.X.isel(**{self.x_sample_dim: slice(None, top_boundary)})
            x_test = xr.concat([ x_bottom_of_window_high_indices, x_top_of_window_low_indices], self.x_sample_dim)
            x_train = self.X.isel(**{self.x_sample_dim: slice(top_boundary, bottom_boundary)})

            y_bottom_of_window_high_indices = self.Y.isel(**{self.y_sample_dim: slice(bottom_boundary, None)})
            y_top_of_window_low_indices = self.Y.isel(**{self.y_sample_dim: slice(None, top_boundary)})
            y_test = xr.concat([ y_bottom_of_window_high_indices, y_top_of_window_low_indices], self.y_sample_dim)
            y_train = self.Y.isel(**{self.y_sample_dim: slice(top_boundary, bottom_boundary)})
        else: 
            x_train_bottom = self.X.isel(**{self.x_sample_dim: slice( None, bottom_boundary )})
            x_train_top = self.X.isel(**{self.x_sample_dim: slice(top_boundary, None )})
            x_train = xr.concat([ x_train_bottom, x_train_top ], self.x_sample_dim)
            x_test= self.X.isel(**{self.x_sample_dim: slice(bottom_boundary, top_boundary)})
            
            y_train_bottom = self.Y.isel(**{self.y_sample_dim: slice( None, bottom_boundary )})
            y_train_top = self.Y.isel(**{self.y_sample_dim: slice(top_boundary, None )})
            y_train = xr.concat([ y_train_bottom, y_train_top ], self.y_sample_dim)
            y_test= self.Y.isel(**{self.y_sample_dim: slice(bottom_boundary, top_boundary)})
        self.index += self.step
        return x_train, y_train, x_test, y_test 

    def __iter__(self):
        return self 
    
    def __next__(self): 
        return self.get_next()
