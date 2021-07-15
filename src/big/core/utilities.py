from .data_validation import *
import numpy as np
import xarray as xr
import datetime as dt

def inner_join(X, Y, dim, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
	assert dim in ['X', 'Y', 'T', 'M'], 'dim must be one of ["X", "Y", "T", "M"]'
	X = standardize_dims(X, x_coords, verbose=verbose)
	Y = standardize_dims(Y, y_coords, verbose=verbose)
	X = X.dropna(x_coords[dim], how="all")
	Y = Y.dropna(y_coords[dim], how="all")

	drop_vals = [val for val in X.coords[x_coords[dim]].values if val not in Y.coords[y_coords[dim]].values ]
	X = X.drop(drop_vals, dim=x_coords[dim])

	drop_vals = [val for val in Y.coords[y_coords[dim]].values if val not in X.coords[x_coords[dim]].values ]
	Y = Y.drop(drop_vals, dim=y_coords[dim])
	return X, Y

### BASH Utilities
def rmrf(dirn):
    subfiles = [file for file in dirn.glob('*') if file.is_file()]
    subdirs = [diro for diro in dirn.glob('*') if diro.is_dir()]

    for file in subfiles:
        file.unlink()
    for subdir in subdirs:
        try:
            subdir.rmdir()
        except:
            rmrf(subdir)
    dirn.rmdir()
