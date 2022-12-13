import scipy.stats as ss
from ..core.utilities import *
import numpy as np
import dask.array as da
from .onehot import quantile

def drymask(X, dry_threshold=0.001, quantile_threshold=0.33, method='midpoint', x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    if X.dims[0] != x_lat_dim or X.dims[1] != x_lon_dim or X.dims[2] != x_sample_dim or X.dims[3] != x_feature_dim:
        X = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    mask2 = X.mean(x_feature_dim).mean(x_sample_dim)
    mask2 = xr.ones_like(mask2).where(~np.isnan(mask2), other=0) # 1 over present, 0 over missing

    bn = quantile(X, quantile_threshold, method=method, x_lat_dim=x_lat_dim, x_lon_dim=x_lon_dim, x_sample_dim=x_sample_dim, x_feature_dim=x_feature_dim)
    mask = bn.where(bn < dry_threshold, other=np.nan) # quant where dry, nan where not
    mask = mask + mask2
    return xr.ones_like(mask).where(np.isnan(mask), other=np.nan).mean(x_feature_dim)
