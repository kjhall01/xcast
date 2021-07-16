from ..core import *
from .methods import *
from ..verification import *
import datetime as dt


def validate(mme, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, method='cross', window=3, initial_training_period=None, training_period_step=None, nd=1, missing_value=-1, fill='mean', save_stepwise_hindcasts=False, save_stepwise_skill=False, verbose=0,  pca_x=False,  n_components=5, rescale_x=None, rescale_y=None, regrid_to='Y', **kwargs):
	assert mme in mme_codes.keys(), '{} Invalid MME type {} - for MultiMME, use MultiMME.validate()'.format(dt.datetime.now(), mme)
	if 'M' not in x_coords.keys():
		x_coords['M'] = 'M'
	if 'M' not in y_coords.keys():
		y_coords['M'] = 'M'
	X = standardize_dims(X, x_coords, verbose=verbose)
	Y = standardize_dims(Y, y_coords, verbose=verbose)

	assert str(regrid_to).upper() in ['X', 'Y'], '{} Must specify regrid_to as either "X" or "Y"'.format(dt.datetime.now())
	if str(regrid_to).upper() == 'Y':
		X, Y = regrid_fill(X, Y, x_coords, y_coords, missing_value=missing_value, verbose=verbose, fill=fill)
	else:
		Y, X = regrid_fill(Y, X, y_coords, x_coords, missing_value=missing_value, verbose=verbose, fill=fill)


	assert method in ['cross', 'retroactive'], '{} invalid validation method {}'.format(dt.datetime.now(), method)
	if method == 'cross':
		x_validator = CrossValidator(X, x_coords=x_coords, window=window)
		y_validator = CrossValidator(Y, x_coords=y_coords, window=window)
	else:
		x_validator = RetroactiveValidator(X, x_coords=x_coords,  initial_training_period=initial_training_period, training_period_step=training_period_step)
		y_validator = RetroactiveValidator(Y, x_coords=y_coords,  initial_training_period=initial_training_period, training_period_step=training_period_step)
	#assert len(x_train) == len(y_train), '{} Mismatched XVAL train length- {} on X, {} on Y'.format(dt.datetime.now(), len(x_train), len(y_train))
	#assert len(x_test) == len(y_test), '{} Mismatched XVAL test length- {} on X, {} on Y'.format(dt.datetime.now(), len(x_test, len(y_test)))

	count=0
	if verbose == 1:
		total = validator.windows * nd
		print('{} {} Validation for {}: ['.format(dt.datetime.now(), method.upper(), mme) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

	if mme in ['SPCR', 'SM_PCR']:
		if 'n_components' in kwargs.keys():
			if kwargs['n_components'] > window:
				kwargs['n_components'] = window
				if verbose:
					print('{} Cannot use n_components < xval window size for SPCR or SM_PCR, setting n_components = window')


	scaler_x, scaler_y = None, None
	pca_x = None
	hindcasts = []
	x_train, x_test = x_validator.get_next_window()
	y_train, y_test = y_validator.get_next_window()
	i=0
	while x_train is not None and y_train is not None:
		i += 1
		if verbose > 1:
			print('\n{} -------------Start Validation Window {} -------------'.format(dt.datetime.now(), i))

		hindcasts.append( [])
		if 'Prob' not in mme:
			if str(rescale_y).upper() == 'NORMAL':
				scaler_y = NormalScaler(**kwargs)
				scaler_y.fit(Y, y_coords, verbose=verbose)
				y_train = scaler_y.transform(y_train, y_coords, verbose=verbose)
				y_test = scaler_y.transform(y_test, y_coords, verbose=verbose)
			elif str(rescale_y).upper() == 'MINMAX':
				scaler_y = MinMaxScaler(**kwargs)
				scaler_y.fit(Y, y_coords, verbose=verbose)
				y_train = scaler_y.transform(y_train, y_coords, verbose=verbose)
				y_test = scaler_y.transform(y_test, y_coords, verbose=verbose)
			else:
				pass
		else:
			#scaler_y = ProbabilisticScaler(**kwargs)
			#scaler_y.fit(Y, y_coords, verbose=verbose)
			#y_train = scaler_y.transform(y_train, y_coords, verbose=verbose)
			#y_test = scaler_y.transform(y_test, y_coords, verbose=verbose)
			y_coords['C'] = 'C'
			pass

		if str(rescale_x).upper() == 'NORMAL':
			scaler_x = NormalScaler(**kwargs)
			scaler_x.fit(X, x_coords, verbose=verbose)
			x_train = scaler_x.transform(x_train, x_coords, verbose=verbose)
			x_test = scaler_x.transform(x_test, x_coords, verbose=verbose)
		elif str(rescale_x).upper() == 'MINMAX':
			scaler_x = MinMaxScaler(**kwargs)
			scaler_x.fit(X, x_coords, verbose=verbose)
			x_train = scaler_x.transform(x_train, x_coords, verbose=verbose)
			x_test = scaler_x.transform(x_test, x_coords, verbose=verbose)
		else:
			pass

		if pca_x:
			pca_x = PrincipalComponents(**kwargs)
			pca_x.fit(x_train, x_coords, verbose=verbose)
			x_train = pca_x.transform(x_train, x_coords, verbose=verbose)
			x_test = pca_x.transform(x_test, x_coords, verbose=verbose)

		for j in range(nd):
			count += 1
			if verbose == 1:
				total = len(x_train) * nd
				print('{} {} Validation for {}: ['.format(dt.datetime.now(), method.upper(), mme) + '*'*int(25 * count / total) + ' '*int(25 * (1 - (count/total))) +'] {}% ({}/{})'.format(int(count/total*100), count, total) , end='\r')

			val_mme = mme_codes[mme](**kwargs)
			val_mme.fit(x_train, y_train, x_coords=x_coords, y_coords=y_coords, rescale_x='NONE', rescale_y='NONE', pca_x=False, verbose=verbose)
			#train = val_mme.predict(x_train, x_coords=x_coords, verbose=verbose)
			if 'Prob' not in mme:
				val = val_mme.predict(x_test, x_coords=x_coords, verbose=verbose)
				if scaler_y is not None:
				#	train = scaler_y.inverse_transform(train, x_coords, verbose=verbose)
					val = scaler_y.inverse_transform(val, x_coords, verbose=verbose)
			else:
				val = val_mme.predict(x_test, x_coords=x_coords, verbose=verbose)


			varname = [iii for iii in val.data_vars][0]
			hindcasts[i-1].append(getattr(val, varname).values)
		hindcasts[i - 1] = np.asarray(hindcasts[i-1])
		x_train, x_test = x_validator.get_next_window()
		y_train, y_test = y_validator.get_next_window()
	if verbose == 1:
		print('{} {} Validation for {}: ['.format(dt.datetime.now(), method.upper(), mme) + '*'*25 + '] 100% ({}/{})'.format( total, total) )

	if 'Prob' not in mme:
		hindcasts = np.concatenate(hindcasts, axis=-1)

		coords = {
			x_coords['X']:X.coords[x_coords['X']].values,
			x_coords['Y']:X.coords[x_coords['Y']].values,
			x_coords['T']:X.coords[x_coords['T']].values,
		}

		x_coords_slice = {'X': x_coords['X'], 'Y': x_coords['Y'], 'T':x_coords['T']}
		y_coords_slice = {'X': y_coords['X'], 'Y': y_coords['Y'], 'T':y_coords['T']}

		iseldict = {x_coords['M']: 0}
		preds = xr.Dataset({'predictions': ([ x_coords['Y'], x_coords['X'], x_coords['T']], np.nanmean(hindcasts, axis=0))}, coords=coords)
		ioa = index_of_agreement(preds, Y,  x_coords=x_coords_slice, y_coords=y_coords_slice, verbose=verbose).skill_measure.isel(**iseldict)
		pearson_coef = pearson_coefficient(preds, Y,  x_coords=x_coords_slice, y_coords=y_coords_slice, verbose=verbose).skill_measure.isel(**iseldict)
		pearson_p = pearson_significance(preds, Y,  x_coords=x_coords_slice, y_coords=y_coords_slice, verbose=verbose).skill_measure.isel(**iseldict)
		spearman_coef = spearman_coefficient(preds, Y,  x_coords=x_coords_slice, y_coords=y_coords_slice, verbose=verbose).skill_measure.isel(**iseldict)
		spearman_p = spearman_significance(preds, Y,  x_coords=x_coords_slice, y_coords=y_coords_slice, verbose=verbose).skill_measure.isel(**iseldict)
		rmse = root_mean_squared_error(preds, Y,  x_coords=x_coords_slice, y_coords=y_coords_slice, verbose=verbose).skill_measure.isel(**iseldict)
		mse = mean_squared_error(preds, Y,  x_coords=x_coords_slice, y_coords=y_coords_slice, verbose=verbose).skill_measure.isel(**iseldict)
		mae = mean_absolute_error(preds, Y, x_coords=x_coords_slice, y_coords=y_coords_slice, verbose=verbose).skill_measure.isel(**iseldict)

		data_vars = {
			'hindcasts': ([x_coords['Y'], x_coords['X'], x_coords['T']], np.nanmean(hindcasts, axis=0)), # axis = 1 takes mean over ND
			'index_of_agreement': ([x_coords['Y'], x_coords['X']], ioa),
			'pearson_coefficient': ([x_coords['Y'], x_coords['X']], pearson_coef),
			'pearson_significance': ([x_coords['Y'], x_coords['X']], pearson_p),
			'spearman_coefficient': ([x_coords['Y'], x_coords['X']], spearman_coef),
			'spearman_significance': ([x_coords['Y'], x_coords['X']], spearman_p),
			'mean_absolute_error': ([x_coords['Y'], x_coords['X']], mae),
			'mean_squared_error': ([x_coords['Y'], x_coords['X']], mse),
			'root_mean_squared_error': ([x_coords['Y'], x_coords['X']], rmse)
		}

	else:
		hindcasts = np.concatenate(hindcasts, axis=-2)

		coords = {
			'C': [0, 1, 2],
			x_coords['X']:X.coords[x_coords['X']].values,
			x_coords['Y']:X.coords[x_coords['Y']].values,
			x_coords['T']:X.coords[x_coords['T']].values,
		}

		x_coords_slice = {'X': x_coords['X'], 'Y': x_coords['Y'], 'C':'C', 'T':x_coords['T']}
		y_coords_slice = {'X': y_coords['X'], 'Y': y_coords['Y'], 'C':'C', 'T':y_coords['T']}

		iseldict = {x_coords['M']: 0}
		print()
		scaler_y = ProbabilisticScaler(**kwargs)
		scaler_y.fit(Y, y_coords, verbose=verbose)
		Y2 = scaler_y.transform(Y, y_coords, verbose=verbose)
		preds = xr.Dataset({'predictions': ([ x_coords['Y'], x_coords['X'], x_coords['T'], 'C' ], np.nanmean(hindcasts, axis=0))}, coords=coords)
		prec = precision(preds, Y2, x_coords=x_coords_slice, y_coords=y_coords_slice,  verbose=verbose).skill_measure
		roc = area_under_roc(preds, Y2, x_coords=x_coords_slice, y_coords=y_coords_slice,  verbose=verbose).skill_measure
		#prec = precision(preds, Y, x_coords=x_coords_slice, y_coords=y_coords_slice,  verbose=False).skill_measure

		data_vars = {
			'hindcasts': ([x_coords['Y'], x_coords['X'], x_coords['T'], 'C'], np.nanmean(hindcasts, axis=0)), # axis = 1 takes mean over ND
			'area_under_roc': ([x_coords['Y'], x_coords['X'], 'C'], roc),
			'precision': ([x_coords['Y'], x_coords['X'], 'C'], prec),
		}


	return xr.Dataset(data_vars, coords=coords)



class CrossValidator:
	def __init__(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, window=3 ):
		self.window = window + 1 if window % 2 == 0 else window
		self.width = int(window // 2)
		self.ndx = self.width
		self.X = standardize_dims(X, x_coords)
		self.x_coords = x_coords
		self.t_size = len(self.X.coords[x_coords['T']].values)
		self.windows = self.t_size
		self.done = False

	def get_next_window(self, verbose=False):
		if self.done:
			return None, None
		range_top = (self.ndx + self.width) % self.t_size
		if ( self.ndx + self.width ) >= self.t_size:
			range_top = self.t_size -1
			self.done =True
		range_bottom = (self.ndx - self.width) % self.t_size
		if range_top > range_bottom:
			t_ndcs = xr.DataArray([j for j in range(self.t_size) if not j >= range_bottom or not j <= range_top], dims=[self.x_coords['T']])
		else:
			t_ndcs = xr.DataArray([j for j in range(self.t_size) if not j >= range_bottom and not j <= range_top], dims=[self.x_coords['T']])
		sel_dict = {self.x_coords['T']: t_ndcs}
		train = self.X.isel(**sel_dict)

		if range_top > range_bottom:
			t_ndcs_test = xr.DataArray([j for j in range(self.t_size) if  j >= range_bottom and j <= range_top], dims=[self.x_coords['T']])
		else:
			t_ndcs_test = xr.DataArray([j for j in range(self.t_size) if  j >= range_bottom or  j <= range_top], dims=[self.x_coords['T']])
		test_sel_dict = {self.x_coords['T']: t_ndcs_test}
		test = self.X.isel(**test_sel_dict)
		self.ndx += self.window
		return train, test


class RetroactiveValidator:
	def __init__(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, initial_training_period=None, training_period_step=None):
		self.X = standardize_dims(X, x_coords)
		self.x_coords = x_coords
		self.t_size = len(self.X.coords[x_coords['T']].values)

		if initial_training_period is None:
			initial_training_period = int(t_size / 10)
		else:
			assert initial_training_period < t_size, 'initial_training_period > total T size'

		if training_period_step is None:
			training_period_step = int(t_size / 10)
		else:
			assert training_period_step < t_size, 'training_period_step > total T size'
		self.ndx = initial_training_period
		self.tps = training_period_step
		self.windows = int((self.t_size - initial_training_period) / self.tps)

	def get_next_window(self, verbose=False):
		if self.ndx >= self.t_size:
			return None, None
		seldict_train, seldict_test = {self.x_coords['T']: slice(None, self.ndx)}, {self.x_coords['T']: slice(self.ndx, None)}
		train_data = X.isel(**seldict_train)
		test_data = X.isel(**seldict_test)
		self.ndx += self.tps
		return train_data, test_data



def cross_validation_split(X, coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, window=3, verbose=False):
	X = standardize_dims(X, coords, verbose=verbose)
	t_size = len(X.coords[coords['T']].values)
	while window > t_size // 2:
		window -= 1
	window = window + 1 if window % 2 == 0 else window
	width = int(window // 2)

	train_data, test_data = [], []
	for i in range(t_size):
		range_top = (i + width) % t_size
		range_bottom = (i - width) % t_size
		if range_top > range_bottom:
			t_ndcs = xr.DataArray([j for j in range(t_size) if not j >= range_bottom or not j <= range_top], dims=[coords['T']])
		else:
			t_ndcs = xr.DataArray([j for j in range(t_size) if not j >= range_bottom and not j <= range_top], dims=[coords['T']])
		sel_dict = {coords['T']: t_ndcs}
		train_data.append(X.isel(**sel_dict))

		if range_top > range_bottom:
			t_ndcs_test = xr.DataArray([j for j in range(t_size) if  j >= range_bottom and j <= range_top], dims=[coords['T']])
		else:
			t_ndcs_test = xr.DataArray([j for j in range(t_size) if  j >= range_bottom or  j <= range_top], dims=[coords['T']])
		test_sel_dict = {coords['T']: t_ndcs_test}
		test_data.append(X.isel(**test_sel_dict))
	return train_data, test_data

def retroactive_split(X, coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, initial_training_period=None, training_period_step=None, verbose=False):
	X = standardize_dims(X, coords, verbose=verbose)
	t_size = len(X.coords[coords['T']].values)
	if initial_training_period is None:
		initial_training_period = int(t_size / 10)
	else:
		assert initial_training_period < t_size, 'initial_training_period > total T size'

	if training_period_step is None:
		training_period_step = int(t_size / 10)
	else:
		assert training_period_step < t_size, 'training_period_step > total T size'

	train_data, test_data = [], []
	for i in range(initial_training_period, t_size, training_period_step):
		seldict_train, seldict_test = {coords['T']: slice(None, i)}, {coords['T']: slice(i, None)}
		train_data.append(X.isel(**seldict_train))
		test_data.append(X.isel(**seldict_test))
	return train_data, test_data
