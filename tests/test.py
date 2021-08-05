import src as xc
import xarray as xr
from pathlib import Path

x = xr.open_dataset('x.nc', decode_times=False)
y = xr.open_dataset('y.nc', decode_times=False)

print('TESTING DETERMINISTIC')
deterministic = [xc.ExtremeLearningMachinePCA, xc.EnsembleMean, xc.ExtremeLearningMachine, xc.BiasCorrectedEnsembleMean, xc.MultipleLinearRegression, xc.PoissonRegression, xc.GammaRegression, xc.FeaturePCR,  xc.RidgeRegressor] # xc.RandomForest, xc.MultiLayerPerceptron
for mme in deterministic:
	if not Path('test/deterministic/{}.nc'.format(mme.__name__)).is_file():
		xval = xc.cross_validate(mme, x.isel(S=slice(None, -7)).prec, y.prate, x_sample_dim='S', window=10, verbose=True)
		xval.to_netcdf('test/deterministic/{}.nc'.format(mme.__name__))
		skill = xc.deterministic_skill(xval.hindcasts, y.prate, x_sample_dim='S')
		skill.to_netcdf('test/deterministic/{}_skill.nc'.format(mme.__name__))

print('SUCCESS')
print('\nTESTING PROBABILISTIC')
probabilistic = [xc.MemberCount,xc.ProbabilisticELMPCA, xc.BiasCorrectedMemberCount, xc.ProbabilisticELM, xc.ProbabilisticNB,   xc.MultiExtendedLogisticRegression, xc.ExtendedLogisticRegression ] #xc.ProbabilisticMLP, xc.ProbabilisticRF,
for mme in probabilistic:
	if not Path('test/probabilistic/{}.nc'.format(mme.__name__)).is_file():
		xval = xc.cross_validate(mme, x.isel(S=slice(None, -7)).prec, y.prate, x_sample_dim='S', window=10, bn_thresh=0.33, an_thresh=0.67, verbose=True)
		xval.to_netcdf('test/probabilistic/{}.nc'.format(mme.__name__))
		skill = xc.probabilistic_skill(xval.hindcasts, y.prate, x_sample_dim='S', x_feature_dim='C')
		skill.to_netcdf('test/probabilistic/{}_skill.nc'.format(mme.__name__))
