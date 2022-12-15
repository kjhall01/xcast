import src as xc
import xarray as xr
import matplotlib.pyplot as plt
import cptcore as cc
import cartopy.crs as ccrs
import numpy as np
import numpy as np
from sklearn.model_selection import KFold
import datetime as dt
import itertools


X, Y = cc.load_southasia_nmme()
Y = Y.expand_dims({'M':[0]})
X = X.expand_dims({'M':[0]})
ohc = xc.RankedTerciles()
ohc.fit(Y)
T = ohc.transform(Y)

y = Y.sel(X=75, Y=10, method='nearest').values.T
x = X.sel(X=75, Y=10, method='nearest').values.T

ht = {
    'save_y': [True],
    'standardize_y': [True],
    'c': [ 3],
    'hidden_layer_size': [ 5,],
    'n_estimators': [50],
    'activation': ['relu', ],
    'preprocessing': ['minmax', ],
    'encoding': [ 'nonexceedance'],
    'quantiles': [  [0.2, 0.4, 0.6, 0.8] ]
}

params, score, score2 = xc.exp.tune(x, y, verbosity=2, params=ht)
print()
print('best_params: ', params)
print('best score: ', score)
print('associated old style score: ', score2)
#score = xc.exp.get_crpss({'save_y': True, 'quantiles': [0.2, 0.4, 0.6, 0.8] }, x, y)



probs, preds = [], []
quants, threshs = [], []
pdfs = []
i=0
for x_train, y_train, x_test, y_test in xc.CrossValidator(X, Y):
    print(i)
    i += 1
    elm = xc.exp.rEPOELM(**params)
    elm.fit(x_train, y_train)
    pd = elm.predict(x_test)
    pdf = elm.predict(x_test, quantile=[0.1, 0.2, 0.5, 0.8, 0.9])
    pr = elm.predict_proba(x_test)
    pq = elm.predict_proba(x_test, quantile=0.2)#[0.2, 0.8])
    pt = elm.predict_proba(x_test, threshold=500)#[500, 900, 1100, 2000])
    probs.append(pr)
    preds.append(pd)
    quants.append(pq)
    threshs.append(pt)
    pdfs.append(pdf)

probs = xr.concat(probs, 'T').mean('ND')
pdfs = xr.concat(pdfs, 'T').mean('ND')

preds = xr.concat(preds, 'T').mean('ND')
quants = xr.concat(quants, 'T').mean('ND')
threshs = xr.concat(threshs, 'T').mean('ND')

probs = xc.gaussian_smooth(probs, kernel=3)
preds = xc.gaussian_smooth(preds, kernel=3)

Y.sel(X=75, Y=10, method='nearest').plot.line(x='T', label='obs')
pdfs.sel(X=75, Y=10, method='nearest').plot.line(x='T', hue='M')
plt.legend()
plt.show()

pearson = xc.Pearson(preds, Y).mean('SKILLDIM').mean('M')
ioa = xc.IndexOfAgreement(preds, Y).mean('SKILLDIM').mean('M')

pe = xc.view(pearson, cmap='RdBu', ocean=False, label='test')
plt.show()

io = xc.view(ioa, vmin=0, cmap='Greens', ocean=False, label='test')
plt.show()


groc = xc.GeneralizedROC(probs, T).mean('SKILLDIM').mean('M')

grel = xc.GREL(probs, T).mean('SKILLDIM').mean('M')


rps = xc.RankProbabilityScore(probs, T).mean('SKILLDIM').mean('M')
clim_rps = xc.RankProbabilityScore(xr.ones_like(probs) / 3.0, T).mean('SKILLDIM').mean('M')

rpss = 1 - rps / clim_rps

import numpy as np

cmap = plt.get_cmap('autumn_r').copy()
cmap.set_under('lightgray')

gr = xc.view(groc, vmin=0.5, cmap=cmap, ocean=False, label='GROC')
plt.show()

gr = xc.view(grel, cmap=cmap, vmin=0.784, ocean=False, label='GREL')
plt.show()

rp = xc.view(rpss, vmin=0, cmap=cmap, label='RPSS', ocean=False)
plt.show()


#probs = probs.where(groc.mean('SKILLDIM').mean('M') > 0.5, other=np.nan)
xc.view_reliability(probs, T, tercile_skill_area=True)
plt.show()

xc.view_probabilistic(probs.isel(T=-1), label='XCAST SAMPLE', savefig='test.png')
plt.show()


xc.view_roc(probs, T)
plt.show()

#pca = xc.PrincipalComponentsAnalysis()
#pca.fit(X)
#M = pca.transform(X)


#elr = xc.cExtendedLogisticRegression()
#elr.fit(X.isel(M=slice(0, 1)), Y)
# pdf = elr.predict_proba(X.isel(M=slice(0, 1)), quantile=[
#                       0.2, 0.3, 0.4, 0.5], n_out=4)


#elm = xc.rExtremeLearningMachine()
#elm.fit(X, T.mean('M'))
#preds = elm.predict(X, n_out=3)
