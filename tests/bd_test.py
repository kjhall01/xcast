import src as xc
import xarray as xr
import numpy as np

bd = xc.open_CsvDataset('bd_varnames_no_tlabels.csv')
y = bd.isel(M=0).expand_dims({'M':[0]})
x = bd.isel(M=slice(-7,None))

results = None
print('TESTING DETERMINISTIC')
deterministic = [xc.EnsembleMean, xc.ExtremeLearningMachine, xc.BiasCorrectedEnsembleMean, xc.MultipleLinearRegression, xc.PoissonRegression, xc.GammaRegression, xc.FeaturePCR, xc.RandomForest, xc.MultiLayerPerceptron, xc.RidgeRegressor] # xc.RandomForest, xc.MultiLayerPerceptron
for mme in deterministic:
	xval = xc.cross_validate(mme, x.climate_var, y.climate_var,  window=1, verbose=True, ND=10)
	if results is None:
		results = xval.hindcasts.isel(X=0,Y=0).transpose('T', 'M').values
	else:
		results = np.hstack([results, xval.hindcasts.isel(X=0,Y=0).transpose('T', 'M').values])
	skill = xc.deterministic_skill(xval.hindcasts, y.climate_var)
np.savetxt('deterministic_bd.csv', results, delimiter=',')

results=None
print('SUCCESS')
print('\nTESTING PROBABILISTIC')
probabilistic = [xc.MemberCount, xc.BiasaCorrectedMemberCount, xc.ProbabilisticELM, xc.ProbabilisticNB,   xc.MultiExtendedLogisticRegression, xc.ExtendedLogisticRegression, xc.ProbabilisticMLP, xc.ProbabilisticRF,] #xc.ProbabilisticMLP, xc.ProbabilisticRF,
for mme in probabilistic:
	xval = xc.cross_validate(mme, x.climate_var, y.climate_var,  window=1, bn_thresh=0.33, an_thresh=0.67, verbose=True, ND=10)
	skill = xc.probabilistic_skill(xval.hindcasts, y.climate_var, x_feature_dim='C')
	if results is None:
		results = xval.hindcasts.isel(X=0,Y=0).transpose('T', 'C').values
	else:
		results = np.hstack([results, xval.hindcasts.isel(X=0,Y=0).transpose('T', 'C').values])
np.savetxt('probabilistic_bd.csv', results, delimiter=',')
print('SUCCESS')
