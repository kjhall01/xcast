from .individual_wrappers import *
from .pointwise_mme import *
import copy
from ..downscaling import *

class ExtremeLearningMachine:
	def __init__(self, hidden_layer_neurons=5, activation='sigm', **kwargs):
		self.kwargs = kwargs
		self.kwargs['hidden_layer_neurons'] = hidden_layer_neurons
		self.kwargs['activation'] = activation
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, ELM_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class ExtremeLearningMachineProba:
	def __init__(self, hidden_layer_neurons=5, activation='sigm', **kwargs):
		self.kwargs = kwargs
		self.kwargs['hidden_layer_neurons'] = hidden_layer_neurons
		self.kwargs['activation'] = activation
		self.model = ProbabilisticPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, ELM_Prob_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill, one_hot=True)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class ELM_PCA:
	def __init__(self, hidden_layer_neurons=5, activation='sigm', n_components=1, **kwargs):
		self.kwargs = kwargs
		self.kwargs['hidden_layer_neurons'] = hidden_layer_neurons
		self.kwargs['activation'] = activation
		self.kwargs['n_components'] = n_components
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=True, fill='mean'):
		self.model.fit(X, Y, ELM_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class ELM_PCA_Proba:
	def __init__(self, hidden_layer_neurons=5, activation='sigm', n_components=1, **kwargs):
		self.kwargs = kwargs
		self.kwargs['hidden_layer_neurons'] = hidden_layer_neurons
		self.kwargs['activation'] = activation
		self.kwargs['n_components'] = n_components
		self.model = ProbabilisticPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=True, fill='mean'):
		self.model.fit(X, Y, ELM_Prob_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill, one_hot=True)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class EnsembleMean:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NONE', rescale_y='NONE', pca_x=False, fill='mean'):
		self.model.fit(X, Y, EnsembleMean_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class BiasCorrectedEnsembleMean:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, EnsembleMean_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class MultipleLinearRegression:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, LinearRegression_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class ModelPCR:
	def __init__(self, n_components=1,  **kwargs):
		self.kwargs = kwargs
		self.kwargs['n_components'] = n_components
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=True, fill='mean'):
		self.model.fit(X, Y, LinearRegression_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class SpatialPCR:
	def __init__(self, n_components=1,  **kwargs):
		self.kwargs = kwargs
		self.kwargs['n_components'] = n_components
		self.n_components = n_components
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='None', rescale_y='NORMAL', pca_x=True, fill='mean'):
		self.spatial_pca = SpatialPrincipalComponents(n_components=self.n_components)
		self.spatial_pca.fit(X, x_coords)
		X = self.spatial_pca.transform(X, x_coords)
		x_varname = [i for i in X.data_vars][0]
		x_vals = []
		for i in range(len(X.coords[x_coords['M']].values)):
			for j in range(len(X.coords['MODE'])):
				iseldict = {x_coords['M']: i, "MODE":j}
				x_vals.append(getattr(X, x_varname).isel(**iseldict).values)
		x_vals = np.asarray(x_vals)
		coords = {
			x_coords['X']: X.coords[x_coords['X']].values,
			x_coords['Y']: X.coords[x_coords['Y']].values,
			x_coords['T']: X.coords[x_coords['T']].values,
		}
		if 'M' in x_coords.keys():
			coords[x_coords['M']] = [i for i in range(len(X.coords[x_coords['M']].values) * len(X.coords['MODE'].values)) ]
			m = x_coords['M']
		else:
			coords['M'] = [i for i in range(len(X.coords[x_coords['M']].values) * len(X.coords['MODE'].values)) ]
			m = 'M'
			x_coords['M'] = 'M'
		data_vars = {'unzipped': ([m, x_coords['Y'], x_coords['X'], x_coords['T']], x_vals)}
		X = xr.Dataset(data_vars, coords=coords)
		self.model.fit(X, Y, LinearRegression_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		self.spatial_pca = SpatialPrincipalComponents(n_components=self.n_components)
		self.spatial_pca.fit(X, x_coords)
		X = self.spatial_pca.transform(X, x_coords)
		x_varname = [i for i in X.data_vars][0]
		x_vals = []
		for i in range(len(X.coords[x_coords['M']].values)):
			for j in range(len(X.coords['MODE'])):
				iseldict = {x_coords['M']: i, "MODE":j}
				x_vals.append(getattr(X, x_varname).isel(**iseldict).values)
		x_vals = np.asarray(x_vals)
		coords = {
			x_coords['X']: X.coords[x_coords['X']].values,
			x_coords['Y']: X.coords[x_coords['Y']].values,
			x_coords['T']: X.coords[x_coords['T']].values,
		}
		if 'M' in x_coords.keys():
			coords[x_coords['M']] = [i for i in range(len(X.coords[x_coords['M']].values) * len(X.coords['MODE'].values)) ]
			m = x_coords['M']
		else:
			coords['M'] = [i for i in range(len(X.coords[x_coords['M']].values) * len(X.coords['MODE'].values)) ]
			m = 'M'
			x_coords['M'] = 'M'
		data_vars = {'unzipped': ([m, x_coords['Y'], x_coords['X'], x_coords['T']], x_vals)}
		X = xr.Dataset(data_vars, coords=coords)
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class SM_PCR:
	def __init__(self, n_components=1,  **kwargs):
		self.kwargs = kwargs
		self.kwargs['n_components'] = n_components
		self.n_components = n_components
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='None', rescale_y='NORMAL', pca_x=True, fill='mean'):
		self.spatial_pca = SpatioModelPrincipalComponents(n_components=self.n_components)
		self.spatial_pca.fit(X, x_coords)
		X = self.spatial_pca.transform(X, x_coords)
		x_coords2 = copy.deepcopy(x_coords)
		x_coords2['M'] = 'MODE'
		self.model.fit(X, Y, LinearRegression_, x_coords=x_coords2, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		X = self.spatial_pca.transform(X, x_coords)
		x_coords2 = copy.deepcopy(x_coords)
		x_coords2['M'] = 'MODE'
		return self.model.predict( X, x_coords=x_coords2, verbose=verbose, fill=fill)



class MultiLayerPerceptron:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, MLPRegressor_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class MultiLayerPerceptronProba:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = ProbabilisticPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, MLPClassifier_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)



class RandomForest:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, RandomForestRegressor_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class RandomForestProba:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = ProbabilisticPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, RandomForestClassifier_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)



class RidgeRegressor:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, Ridge_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

class SingularValueDecomposition:
	def __init__(self, **kwargs):
		self.kwargs = kwargs
		self.model = DeterministicPointWiseMME(**self.kwargs)

	def fit(self, X, Y, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, y_coords={'X':'X', 'Y':'Y', 'T':'T'}, verbose=False, rescale_x='NORMAL', rescale_y='NORMAL', pca_x=False, fill='mean'):
		self.model.fit(X, Y, SVDRegressor_, x_coords=x_coords, y_coords=y_coords, verbose=verbose, rescale_x=rescale_x, rescale_y=rescale_y, pca_x=pca_x, fill=fill)

	def predict(self, X, x_coords={'X':'X', 'Y':'Y', 'T':'T', 'M':'M'}, verbose=False, fill='mean'):
		return self.model.predict( X, x_coords=x_coords, verbose=verbose, fill=fill)

mme_codes = {
	# Deterministic
	'ELM':ExtremeLearningMachine,
	'ELMPCA': ELM_PCA,
	'EM': EnsembleMean,
	'BCEM': BiasCorrectedEnsembleMean,
	'SVD': SingularValueDecomposition,
	'RIDGE': RidgeRegressor,
	'MLP': MultiLayerPerceptron,
	'RF': RandomForest,
	'MLR': MultipleLinearRegression,
	'MPCR': ModelPCR,
	'SPCR': SpatialPCR,
	'SM_PCR': SM_PCR,
	'CNN': ConvolutionalNeuralNetwork,

	# Probabilistic
	'ProbMLP': MultiLayerPerceptronProba,
	'ProbRF': RandomForestProba,
	'ProbCNN': ConvolutionalNeuralNetworkProba,
	'ProbELM':ExtremeLearningMachineProba,
	'ProbELMPCA': ELM_PCA_Proba
}
