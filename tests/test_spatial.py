import src as xc
import xarray as xr
from pathlib import Path

x = xr.open_dataset('x.nc', decode_times=False)
y = xr.open_dataset('y.nc', decode_times=False)

print('TESTING DETERMINISTIC')
deterministic = [  xc.SpatialPCR, xc.SpatialELM, xc.SpatialRidge,  xc.SpatialGammaRegression, xc.SpatialPoissonRegression, xc.SpatialMLP, xc.SpatialRF]
for mme in deterministic:
	if not Path('test_spatial/deterministic/{}.nc'.format(mme.__name__)).is_file():
		xval = xc.cross_validate(mme, x.isel(S=slice(None, -7)).prec, y.prate, x_sample_dim='S', window=10, verbose=True)
		xval.to_netcdf('test_spatial/deterministic/{}.nc'.format(mme.__name__))
		skill = xc.deterministic_skill(xval.hindcasts, y.prate, x_sample_dim='S')
		skill.to_netcdf('test_spatial/deterministic/{}_skill.nc'.format(mme.__name__))

print('SUCCESS')
print('\nTESTING PROBABILISTIC')
probabilistic = [xc.SpatialPOELM,  xc.SpatialMultiELR, xc.SpatialELR, xc.ProbabilisticSpatialMLP, xc.ProbabilisticSpatialRF]
for mme in probabilistic:
	if not Path('test_spatial/probabilistic/{}.nc'.format(mme.__name__)).is_file():
		xval = xc.cross_validate(mme, x.isel(S=slice(None, -7)).prec, y.prate, x_sample_dim='S', window=10, bn_thresh=0.33, an_thresh=0.67, verbose=True)
		xval.to_netcdf('test_spatial/probabilistic/{}.nc'.format(mme.__name__))
		skill = xc.probabilistic_skill(xval.hindcasts, y.prate, x_sample_dim='S', x_feature_dim='C')
		skill.to_netcdf('test_spatial/probabilistic/{}_skill.nc'.format(mme.__name__))
print('SUCCESS')
