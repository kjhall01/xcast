import src as xc
import xarray as xr
import matplotlib.pyplot as plt
import cptcore as cc
import cartopy.crs as ccrs
import numpy as np
import einstein_epoelm as ee



X, Y = cc.load_southasia_nmme()
Y = Y.expand_dims({'M':[0]})
X = X.expand_dims({'M':[0]})

ohc = xc.RankedTerciles()
ohc.fit(Y)
T = ohc.transform(Y)

#xc.view_taylor(X, Y)
#plt.show()

#print(Y)
#print()
#gamm = xc.EmpiricalTransformer()
#gamm.fit(Y)

#print(Y)
#print()
#gam_y = gamm.transform(Y)

#gam2 = xc.EmpiricalTransformer()
#gam2.fit(X)
#gam_x = gam2.transform(X)

#xc.view_taylor(gam_x, gam_y)
#plt.show()

#pe = xc.Pearson(gam_x, gam_y)
#pe.plot()
#plt.show()


probs, preds = [], []
i=0
for x_train, y_train, x_test, y_test in xc.CrossValidator(X, Y):
    print(i)
    i += 1
    elm = ee.rEPOELM()
    elm.fit(x_train, y_train)
    pd = elm.predict(x_test)
    pr = elm.predict_proba(x_test)
    probs.append(pr)
    preds.append(pd)

probs = xr.concat(probs, 'T').mean('ND')
preds = xr.concat(preds, 'T').mean('ND')

probs = xc.gaussian_smooth(probs, kernel=9)
preds = xc.gaussian_smooth(preds, kernel=9)

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
