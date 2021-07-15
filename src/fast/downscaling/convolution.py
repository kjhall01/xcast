import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime as dt
from ..core import *
from .projection import *
from ..preprocessing import *
import xarray as xr

class ConvolutionalNeuralNetwork:
	def __init__(self, kernel_sizes=[3], **kwargs):
		self.kernel_sizes = kernel_sizes
		self.model = None
		self.scaler_x = None
		self.nanmask = None
		self.kwargs=kwargs
		self.scaler_y = None

	def summary(self):
		assert self.model is not None, '{} ConvolutionalNeuralNetwork has no defined model'.format(dt.datetime.now())
		self.model.summary()

	def plot_model(self, filename='cnn.png'):
		assert self.model is not None, '{} ConvolutionalNeuralNetwork has no defined model'.format(dt.datetime.now())
		keras.utils.plot_model(self.model, filename, show_shapes=True)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'S', "M":'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', "M":'M'}, verbose=0, missing_value=-1,  batch_size=64, epochs=10, validation_split=0.1, rescale_x='NORMAL', rescale_y='NONE', pca_x=None,  fill='kernel', mask_nans=True,model=None, loss='mean_squared_logarithmic_error'):
		assert loss in ['mean_squared_logarithmic_error', 'mean_squared_error', 'mean_absolute_error'], '{} Invalid Loss function - {}'.format(dt.datetime.now(), loss)
		assert self.model is None, '{} Cannot Re-Fit a CNN Type'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys() and 'M' in x_coords.keys(), '{} X must have XYTM dimensions - not shown in x_coords'.format(dt.datetime.now())
		assert 'X' in y_coords.keys() and 'Y' in y_coords.keys() and 'T' in y_coords.keys(), '{} Y must have XYT dimensions - not shown in x_coords'.format(dt.datetime.now())
		if 'M' not in y_coords.keys():
			y_coords['M'] = 'M'
		Y = standardize_dims(Y, y_coords, verbose=verbose)
		X = standardize_dims(X, x_coords, verbose=verbose)

		if str(rescale_y).upper() == 'NORMAL':
			self.scaler_y = NormalScaler(**self.kwargs)
			self.scaler_y.fit(Y, y_coords, verbose=verbose)
			Y = self.scaler_y.transform(Y, y_coords, verbose=verbose)
		elif str(rescale_y).upper() == 'MINMAX':
			self.scaler_y = MinMaxScaler(**self.kwargs)
			self.scaler_y.fit(Y, y_coords, verbose=verbose)
			Y = self.scaler_y.transform(Y, y_coords, verbose=verbose)
		else:
			pass

		if str(rescale_x).upper() == 'NORMAL':
			self.scaler_x = NormalScaler(**self.kwargs)
			self.scaler_x.fit(X, x_coords, verbose=verbose)
			X = self.scaler_x.transform(X, x_coords, verbose=verbose)
		elif str(rescale_x).upper() == 'MINMAX':
			self.scaler_x = MinMaxScaler(**kwargs)
			self.scaler_x.fit(X, x_coords, verbose=verbose)
			X = self.scaler_x.transform(X, x_coords, verbose=verbose)
		else:
			pass

		if mask_nans:
			nanmask = get_nanmask(Y, y_coords)
			X = mask_nan(X, x_coords, nanmask=nanmask)
			self.nanmask = nanmask

		xvarname, yvarname = [i for i in X.data_vars][0], [i for i in Y.data_vars][0]
		self.Y, self.y_coords = Y, y_coords
		if model is None:
			input_layer = keras.Input(shape=(len(X.coords[x_coords['Y']].values), len(X.coords[x_coords['X']].values), len(X.coords[x_coords['M']].values)), name="base")
			output_layer = layers.Conv2D(1, self.kernel_sizes[0], activation="relu", padding='same')(input_layer)
			self.model = keras.Model(input_layer, output_layer, name="convolution")
		else:
			self.model = model
		self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1), loss=loss)
		X = X.transpose(x_coords['T'], x_coords['Y'], x_coords['X'], x_coords['M'])
		Y = Y.transpose(y_coords['T'], y_coords['Y'], y_coords['X'], y_coords['M'])
		#fit the model now on (T, Y, X, M) shaped array
		xvarname, yvarname = [i for i in X.data_vars][0], [i for i in Y.data_vars][0]
		x_train, y_train = getattr(X, xvarname).values, getattr(Y, yvarname).values
		verbose = 0 if verbose < 2 else 1
		self.model.fit(x_train, y_train, batch_size=min(batch_size, x_train.shape[0]), epochs=epochs, validation_split=validation_split, verbose=verbose)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'S', "M":'M'}, verbose=0, missing_value=-999, fill='kernel', mask_nans=True):
		assert self.model is not None, '{} Must fit a CNN Type before predict'.format(dt.datetime.now())
		assert 'X' in x_coords.keys() and 'Y' in x_coords.keys() and 'T' in x_coords.keys() and 'M' in x_coords.keys(), '{} X must have XYTM dimensions - not shown in x_coords'.format(dt.datetime.now())

		X = standardize_dims(X, x_coords, verbose=verbose)
		x_coords_slice = {'X':x_coords['X'], 'Y':x_coords['Y']}
		y_coords_slice = {'X':self.y_coords['X'], 'Y':self.y_coords['Y']}

		check_same_shape(X, self.Y, x_coords=x_coords_slice, y_coords=y_coords_slice)

		if self.scaler_x is not None:
			X = self.scaler_x.transform(X, x_coords, verbose=verbose)

		if mask_nans:
			X = mask_nan(X, x_coords, nanmask=self.nanmask)


		X = X.transpose(x_coords['T'], x_coords['Y'], x_coords['X'], x_coords['M'])
		#fit the model now on (T, Y, X, M) shaped array
		xvarname = [i for i in X.data_vars][0]

		x_train = getattr(X, xvarname).values
		preds = self.model.predict(x_train)
		coords = {
			x_coords['M']: [0],
			x_coords['X']: X.coords[x_coords['X']].values,
			x_coords['Y']: X.coords[x_coords['Y']].values,
			x_coords['T']: X.coords[x_coords['T']].values,
		}
		data_vars = {xvarname: ([x_coords['T'], x_coords['Y'], x_coords['X'], x_coords['M'] ], preds)}
		ret =  xr.Dataset(data_vars, coords=coords)
		if self.scaler_y is not None:
			ret = self.scaler_y.inverse_transform(ret, x_coords)
		return ret.mean('M')
