from .pointwise_preprocess import PointWisePreprocess
import numpy as np
import datetime as dt
import uuid
from pathlib import Path
import xarray as xr

class NormalScaler_:
	def __init__(self):
		self.mu, self.std = None, None

	def fit(self, x):
		self.mu = np.nanmean(x)
		self.std = np.nanstd(x)
		self.std = 1 if self.std == 0 else self.std
		self.mu = 0 if self.std == 0 else self.mu

	def transform(self, x):
		assert self.mu is not None and self.std is not None, '{} Must Fit Scaler_ before transform'.format(dt.datetime.now())
		return ((x - self.mu) / self.std)

	def inverse_transform(self, x):
		assert self.mu is not None and self.std is not None, '{} Must Fit Scaler_ before transform'.format(dt.datetime.now())
		return ((x * self.std) + self.mu)


class NormalScaler:
	def __init__(self, **kwargs):
		self.pointwise = []
		self.kwargs = kwargs

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		self.mu = X.mean(x_coords['T'])
		self.std = X.std(x_coords['T'])

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		return (X - self.mu ) / self.std

	def inverse_transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False):
		return (X * self.std) + self.mu

class NormalScalerPointWise:
	def __init__(self, **kwargs):
		self.pointwise = []
		self.kwargs = kwargs

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False, t_chunks=20, x_chunks=10, y_chunks=10):
		if verbose > 1:
			print('{} Starting Normalize Fit'.format(dt.datetime.now()))
		if 'M' in x_coords.keys():
			X = X.chunk({
				x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)),
				x_coords['M']: len(X.coords[x_coords['M']].values),
				x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks)),
				x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks))
			})
			for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
				self.pointwise.append([])
				indx = 0
				for ii in range(i):
					indx += X.chunks[x_coords['Y']][ii]
				for j in range(len(X.chunks[x_coords['X']])): # chunks in X
					print('{} {}'.format(i, j), end='\r')
					jndx = 0
					for jj in range(j):
						jndx += X.chunks[x_coords['X']][jj]
					self.pointwise[i].append(PointWisePreprocess(**self.kwargs))
					iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
					self.pointwise[i][j].fit(X.isel(**iseldict), NormalScaler_, x_coords=x_coords,  verbose=verbose)
		else:
			X = X.chunk({
				x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)),
				x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks)),
				x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks))
			})
			for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
				self.pointwise.append([])
				indx = 0
				for ii in range(i):
					indx += X.chunks[x_coords['Y']][ii]
				for j in range(len(X.chunks[x_coords['X']])): # chunks in X
					jndx = 0
					for jj in range(j):
						jndx += X.chunks[x_coords['X']][jj]
					self.pointwise[i].append(PointWisePreprocess(**self.kwargs))
					iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
					self.pointwise[i][j].fit(X.isel(**iseldict), NormalScaler_, x_coords=x_coords,  verbose=verbose)
		if verbose > 1:
			print('{} Finishing Normalize Fit'.format(dt.datetime.now()))


	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
		if verbose > 1:
			print('{} Starting Normalize Transform'.format(dt.datetime.now()))
		tf_uuid = str(uuid.uuid4())
		ids = []
		if 'M' in x_coords.keys():
			chunks = {
				x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)),
				x_coords['M']: len(X.coords[x_coords['M']].values),
				x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks)),
				x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks))
			}
			X = X.chunk(chunks)
			for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
				indx = 0
				for ii in range(i):
					indx += X.chunks[x_coords['Y']][ii]
				for j in range(len(X.chunks[x_coords['X']])): # chunks in X
					jndx = 0
					for jj in range(j):
						jndx += X.chunks[x_coords['X']][jj]
					iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
					self.pointwise[i][j].transform(X.isel(**iseldict), x_coords=x_coords,  verbose=verbose).to_netcdf(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
					ids.append(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
		else:
			chunks = {
				x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)),
				x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks)),
				x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks))
			}
			X = X.chunk(chunks)
			for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
				indx = 0
				for ii in range(i):
					indx += X.chunks[x_coords['Y']][ii]
				for j in range(len(X.chunks[x_coords['X']])): # chunks in X
					jndx = 0
					for jj in range(j):
						jndx += X.chunks[x_coords['X']][jj]
					iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
					self.pointwise[i][j].transform(X.isel(**iseldict), x_coords=x_coords,  verbose=verbose).to_netcdf(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
					ids.append(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
		if verbose > 1:
			print('{} Finishing Normalize Transform'.format(dt.datetime.now()))
		return xr.open_mfdataset(ids, chunks='auto', parallel=True)

	def inverse_transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
		if verbose > 1:
			print('{} Starting Normalize Inverse Transform'.format(dt.datetime.now()))
		tf_uuid = str(uuid.uuid4())
		ids = []
		if 'M' in x_coords.keys():
			chunks = {
				x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)),
				x_coords['M']: len(X.coords[x_coords['M']].values),
				x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks)),
				x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks))
			}
			X = X.chunk(chunks)
			for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
				indx = 0
				for ii in range(i):
					indx += X.chunks[x_coords['Y']][ii]
				for j in range(len(X.chunks[x_coords['X']])): # chunks in X
					jndx = 0
					for jj in range(j):
						jndx += X.chunks[x_coords['X']][jj]
					iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
					self.pointwise[i][j].inverse_transform(X.isel(**iseldict), x_coords=x_coords,  verbose=verbose).to_netcdf(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
					ids.append(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
		else:
			chunks = {
				x_coords['T']: max(1, int(len(X.coords[x_coords['T']].values) / t_chunks)),
				x_coords['X']: max(1, int(len(X.coords[x_coords['X']].values) / x_chunks)),
				x_coords['Y']: max(1, int(len(X.coords[x_coords['Y']].values) / y_chunks))
			}
			X = X.chunk(chunks)
			for i in range(len(X.chunks[x_coords['Y']])): #chunks in Y
				indx = 0
				for ii in range(i):
					indx += X.chunks[x_coords['Y']][ii]
				for j in range(len(X.chunks[x_coords['X']])): # chunks in X
					jndx = 0
					for jj in range(j):
						jndx += X.chunks[x_coords['X']][jj]
					iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
					self.pointwise[i][j].inverse_transform(X.isel(**iseldict), x_coords=x_coords,  verbose=verbose).to_netcdf(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
					ids.append(Path().home() / '.xcast_cache' / '{}_{}_{}.nc'.format(tf_uuid, i , j))
		if verbose > 1:
			print('{} Finishing Normalize Inverse Transform'.format(dt.datetime.now()))
		return xr.open_mfdataset(ids, chunks='auto', parallel=True)
