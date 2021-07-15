from .pointwise_preprocess import PointWisePreprocess
import numpy as np
import datetime as dt
import xarray as xr
import uuid
from pathlib import Path

class MinMaxScaler_:
	def __init__(self, min=-1, max=1 , **kwargs):
		self.range_min, self.range_max = min, max
		self.range = max - min
		self.min, self.max, self.x_range = None, None, None

	def fit(self, x):
		self.min = np.nanmin(x)
		self.max = np.nanmax(x)
		self.x_range = self.max - self.min

	def transform(self, x):
		assert self.min is not None and self.max is not None, '{} Must Fit MinMaxScaler_ before transform'.format(dt.datetime.now())
		return ((x - self.min) / self.x_range) * self.range + self.range_min

	def inverse_transform(self, x):
		assert self.mu is not None and self.std is not None, '{} Must Fit Scaler_ before transform'.format(dt.datetime.now())
		return ((x - self.range_min) / self.range) * self.x_range + self.min

class MinMaxScaler:
	def __init__(self, min=-1, max=1, **kwargs):
		self.range_min, self.range_max = min, max
		self.range = max - min
		self.min, self.max, self.x_range = None, None, None

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False):
		self.min = X.min(x_coords['T'])
		self.max = X.max(x_coords['T'])
		self.x_range = self.max - self.min

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False):
		return ((X - self.min) / self.x_range) * self.range + self.range_min

	def transform(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False):
		return ((X - self.range_min) / self.range) * self.x_range + self.min

class MinMaxScalerPointwise:
	def __init__(self, min=-1, max=1, **kwargs):
		self.min=min
		self.max= 1
		self.range = max - min
		self.pointwise = []
		self.kwargs = kwargs

	def fit(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'},verbose=False, x_chunks=10, y_chunks=10, t_chunks=20):
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
					jndx = 0
					for jj in range(j):
						jndx += X.chunks[x_coords['X']][jj]
					self.pointwise[i].append(PointWisePreprocess(**self.kwargs))
					iseldict = {x_coords['X']: slice(jndx, jndx+X.chunks[x_coords['X']][j]), x_coords['Y']: slice(indx, indx+X.chunks[x_coords['Y']][i])  }
					self.pointwise[i][j].fit(X.isel(**iseldict), MinMaxScaler_, x_coords=x_coords,  verbose=verbose)
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
					self.pointwise[i][j].fit(X.isel(**iseldict), MinMaxScaler_, x_coords=x_coords,  verbose=verbose)
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
