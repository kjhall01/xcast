import numpy as np
import hpelm
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import datetime as dt

class LinearRegression_:
	def __init__(self, **kwargs):
		args = {}
		args['fit_intercept'] = True if 'fit_intercept' not in kwargs.keys() else kwargs['fit_intercept']
		args['normalize'] = False if 'normalize' not in kwargs.keys() else kwargs['normalize']
		args['copy_X'] = True  if 'copy_X' not in kwargs.keys() else kwargs['copy_X']
		args['n_jobs'] = None  if 'n_jobs' not in kwargs.keys() else kwargs['n_jobs']
		args['positive'] = False  if 'positive' not in kwargs.keys() else kwargs['positive']
		self.model = LinearRegression(**args)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict(self, x):
		return self.model.predict(x)

class MLPRegressor_:
	def __init__(self, **kwargs):
		args = {}
		args['hidden_layer_sizes']=100 if 'hidden_layer_sizes' not in kwargs.keys() else kwargs['hidden_layer_sizes']
		args['activation']='relu' if 'activation' not in kwargs.keys() else kwargs['activation']
		args['solver']='adam' if 'solver' not in kwargs.keys() else kwargs['solver']
		args['alpha']=0.0001 if 'alpha' not in kwargs.keys() else kwargs['alpha']
		args['batch_size']='auto' if 'batch_size' not in kwargs.keys() else kwargs['batch_size']
		args['learning_rate']='constant' if 'learning_rate' not in kwargs.keys() else kwargs['learning_rate']
		args['learning_rate_init']=0.001 if 'learning_rate_init' not in kwargs.keys() else kwargs['learning_rate_init']
		args['power_t']=0.5 if 'power_t' not in kwargs.keys() else kwargs['power_t']
		args['max_iter']=200 if 'max_iter' not in kwargs.keys() else kwargs['max_iter']
		args['shuffle']=True if 'shuffle' not in kwargs.keys() else kwargs['shuffle']
		args['random_state']=None if 'random_state' not in kwargs.keys() else kwargs['random_state']
		args['tol']=0.0001 if 'tol' not in kwargs.keys() else kwargs['tol']
		args['verbose']=False if 'verbose' not in kwargs.keys() else kwargs['verbose']
		args['warm_start']=False if 'warm_start' not in kwargs.keys() else kwargs['warm_start']
		args['momentum']=0.9 if 'momentum' not in kwargs.keys() else kwargs['momentum']
		args['nesterovs_momentum']=True if 'nesterovs_momentum' not in kwargs.keys() else kwargs['nesterovs_momentum']
		args['early_stopping']=False if 'early_stopping' not in kwargs.keys() else kwargs['early_stopping']
		args['validation_fraction']=0.1 if 'validation_fraction' not in kwargs.keys() else kwargs['validation_fraction']
		args['beta_1']=0.9 if 'beta_1' not in kwargs.keys() else kwargs['beta_1']
		args['beta_2']=0.999 if 'beta_2' not in kwargs.keys() else kwargs['beta_2']
		args['epsilon']=0.00000001 if 'epsilon' not in kwargs.keys() else kwargs['epsilon']
		args['n_iter_no_change']=10 if 'n_iter_no_change' not in kwargs.keys() else kwargs['n_iter_no_change']
		args['max_fun']=15000 if 'max_fun' not in kwargs.keys() else kwargs['max_fun']
		self.model = MLPRegressor(**args)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict(self, x):
		return self.model.predict(x)

class RandomForestRegressor_:
	def __init__(self, **kwargs):
		args = {}
		args['n_estimators']=100 if 'n_estimators' not in kwargs.keys() else kwargs['n_estimators']
		args['criterion']='mse' if 'criterion' not in kwargs.keys() else kwargs['criterion']
		args['max_depth']=None if 'max_depth' not in kwargs.keys() else kwargs['max_depth']
		args['min_samples_split']=2 if 'min_samples_split' not in kwargs.keys() else kwargs['min_samples_split']
		args['min_samples_leaf']=1 if 'min_samples_leaf' not in kwargs.keys() else kwargs['min_samples_leaf']
		args['min_weight_fraction_leaf']=0.0 if 'min_weight_fraction_leaf' not in kwargs.keys() else kwargs['min_weight_fraction_leaf']
		args['max_features']='auto' if 'max_features' not in kwargs.keys() else kwargs['max_features']
		args['max_leaf_nodes']=None if 'max_leaf_nodes' not in kwargs.keys() else kwargs['max_leaf_nodes']
		args['min_impurity_decrease']=0.0 if 'min_impurity_decrease' not in kwargs.keys() else kwargs['min_impurity_decrease']
		args['min_impurity_split']=None if 'min_impurity_split' not in kwargs.keys() else kwargs['min_impurity_split']
		args['bootstrap']=True if 'bootstrap' not in kwargs.keys() else kwargs['bootstrap']
		args['oob_score']=False if 'oob_score' not in kwargs.keys() else kwargs['oob_score']
		args['n_jobs']=None if 'n_jobs' not in kwargs.keys() else kwargs['n_jobs']
		args['random_state']=None if 'random_state' not in kwargs.keys() else kwargs['random_state']
		args['verbose']=False if 'verbose' not in kwargs.keys() else kwargs['verbose']
		args['warm_start']=False if 'warm_start' not in kwargs.keys() else kwargs['warm_start']
		args['ccp_alpha']=0.0 if 'ccp_alpha' not in kwargs.keys() else kwargs['ccp_alpha']
		args['max_samples']=None if 'max_samples' not in kwargs.keys() else kwargs['max_samples']
		self.model = RandomForestRegressor(**args)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict(self, x):
		return self.model.predict(x)

class Ridge_:
	def __init__(self, **kwargs):
		args = {}
		args['fit_intercept'] = True if 'fit_intercept' not in kwargs.keys() else kwargs['fit_intercept']
		args['normalize'] = False if 'normalize' not in kwargs.keys() else kwargs['normalize']
		args['copy_X'] = True  if 'copy_X' not in kwargs.keys() else kwargs['copy_X']
		args['alpha']=1.0 if 'alpha' not in kwargs.keys() else kwargs['alpha']
		args['max_iter']=None if 'max_iter' not in kwargs.keys() else kwargs['max_iter']
		args['tol']=0.001 if 'tol' not in kwargs.keys() else kwargs['tol']
		args['solver']='auto' if 'solver' not in kwargs.keys() else kwargs['solver']
		args['random_state']=None if 'random_state' not in kwargs.keys() else kwargs['random_state']
		self.model = Ridge(**args)

	def fit(self, x, y):
		self.model.fit(x, y)

	def predict(self, x):
		return self.model.predict(x)


class ELM_:
	def __init__(self, **kwargs):
		assert 'x_train_shape' in kwargs.keys(), '{} ELM Requires x_train_shape in kwargs'.format(dt.datetime.now())
		assert 'y_train_shape' in kwargs.keys(), '{} ELM Requires y_train_shape in kwargs'.format(dt.datetime.now())
		self.model = hpelm.ELM(kwargs['x_train_shape'], kwargs['y_train_shape'])
		self.model.add_neurons(kwargs['hidden_layer_neurons'], kwargs['activation'])

	def fit(self, x, y):
		self.model.train(x, y, 'r')

	def predict(self, x):
		return self.model.predict(x)


class EnsembleMean_:
	"""Wrapper class for Ensemble Mean"""
	def __init__(self, **kwargs ):
		self.kwargs = kwargs

	def predict(self, x):
		return np.nanmean(x, axis=1).reshape(-1,1)

	def fit(self, x, y):
		return np.nanmean(x, axis=1).reshape(-1,1)

class SVDRegressor_:
	"""Wrapper class for Singular Value Decomposition MLR methodology"""
	def __init__(self, **kwargs):
		self.kwargs = kwargs

	#method from https://machinelearningmastery.com/solve-linear-regression-using-linear-algebra/
	def fit(self, x, y):
		self.b = np.linalg.pinv(x).dot(y)
		return self

	def predict(self, x):
		return x.dot(self.b)
