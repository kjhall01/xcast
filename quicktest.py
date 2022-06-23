import src as xc
import xarray as xr
import matplotlib.pyplot as plt
X, Y, T = xc.NMME_IMD_ISMR()
mlr = xc.cPOELM(lat_chunks=4, lon_chunks=4)
mlr.fit(X, T.mean('M'), rechunk=True)
p = mlr.predict(X, rechunk=True)
p = mlr.predict_proba(X, rechunk=True)


pca = xc.PrincipalComponentsAnalysis()
pca.fit(X)
M = pca.transform(X)


#elr = xc.cExtendedLogisticRegression()
#elr.fit(X.isel(M=slice(0, 1)), Y)
# pdf = elr.predict_proba(X.isel(M=slice(0, 1)), quantile=[
#                       0.2, 0.3, 0.4, 0.5], n_out=4)


elm = xc.rExtremeLearningMachine()
elm.fit(X, T.mean('M'))
preds = elm.predict(X, n_out=3)
