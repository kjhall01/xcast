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
            test_high_indices = [i for i in range(bottom_boundary, self.samples)]
            test_low_indices = [ i for i in range(top_boundary)]
            test_high_indices.extend(test_low_indices)
            
            train_indices = [i for i in range(top_boundary, bottom_boundary)]

            x_test, y_test = self.X.isel(**{self.x_sample_dim: test_high_indices}), self.Y.isel(**{self.y_sample_dim: test_high_indices})
            x_train, y_train = self.X.isel(**{self.x_sample_dim: train_indices}), self.Y.isel(**{self.y_sample_dim: train_indices})

        else: 
            train_low_indices = [i for i in range(bottom_boundary)]
            train_high_indices = [ i for i in range(top_boundary, self.samples)]
            train_low_indices.extend(train_high_indices)
            test_indices = [i for i in range(bottom_boundary, top_boundary)]

            x_test, y_test = self.X.isel(**{self.x_sample_dim: test_indices}), self.Y.isel(**{self.y_sample_dim: test_indices})
            x_train, y_train = self.X.isel(**{self.x_sample_dim: train_low_indices}), self.Y.isel(**{self.y_sample_dim: train_low_indices})

        self.index += self.step
        return x_train, y_train, x_test, y_test 

    def __iter__(self):
        return self 
    
    def __next__(self): 
        return self.get_next()
