from .base_preprocess import *
from sklearn.decomposition import PCA, NMF, FactorAnalysis, DictionaryLearning, IncrementalPCA

class PrincipalComponentsAnalysis(BasePreprocess):
	def __init__(self, use_dask=False, n_components=2, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = PCA
		assert type(n_components) == int, 'we only use n_components as an integer number of modes, rather than a % of variance'
		self.kwargs['n_components'] = n_components


class NMF(BasePreprocess):
	def __init__(self, use_dask=False, n_components=2, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = NMF
		assert type(n_components) == int, 'we only use n_components as an integer number of modes, rather than a % of variance'
		self.kwargs['n_components'] = n_components


class FactorAnalysis(BasePreprocess):
	def __init__(self, use_dask=False, n_components=2, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = FactorAnalysis
		assert type(n_components) == int, 'we only use n_components as an integer number of modes, rather than a % of variance'
		self.kwargs['n_components'] = n_components


class DictionaryLearning(BasePreprocess):
	def __init__(self, use_dask=False, n_components=2, **kwargs):
		super().__init__(use_dask=use_dask, **kwargs)
		self.model_type = DictionaryLearning
		assert type(n_components) == int, 'we only use n_components as an integer number of modes, rather than a % of variance'
		self.kwargs['n_components'] = n_components
