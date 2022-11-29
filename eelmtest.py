import src as xc
import matplotlib.pyplot as plt
import cptcore as cc
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.special import logit

X, Y = cc.load_southasia_nmme()
Y = Y.expand_dims({'M':[0]})
X = X.expand_dims({'M':[0]})

x = X.isel(X=20, Y=20).values.T
y = Y.isel(X=20, Y=20).values.T

reg = LinearRegression()
reg.fit(x, y)
xt = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
preds = reg.predict(xt)
plt.scatter(x, y, color='blue')
plt.scatter(xt, preds, color='red')





for i in [1, 2, 3, 5, 10, 15, 25, 50, 100]:
    elm_kwargs = dict(n_estimators=100, hidden_layer_size=i, activation='tanh', preprocessing='std')
    eelm = xc.EinsteinLearningMachine(**elm_kwargs)
    eelm.fit(x, y)
    epreds = eelm.predict(xt)
    plt.scatter(xt, epreds, color='green', s=0.1)
plt.show()



# one-hot encode Y
bn = np.quantile(y.squeeze(), (1/3.0))
an =  np.quantile(y.squeeze(), (2/3.0))
y_terc  = np.ones((y.shape[0], 3))
y_terc[(y.ravel() > bn) & (y.ravel() <=an), 0] = 0
y_terc[(y.ravel() > bn) & (y.ravel() <=an), 2] = 0
y_terc[y.ravel() < bn, 2] = 0
y_terc[y.ravel() < bn, 1] = 0
y_terc[y.ravel() >= an, 0] = 0
y_terc[y.ravel() >= an, 1] = 0
y_terc -= 0.0001
y_terc = np.abs(y_terc)
y_logs = logit(y_terc)


elm_kwargs = dict(n_estimators=30, hidden_layer_size=5, c=-5, activation='relu', preprocessing='std')
reg1 = xc.EinsteinLearningMachine(**elm_kwargs)
reg1.fit(x, y)
xt = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
preds2 = reg1.predict_noexpit(xt)
preds22 = reg1.predict_noexpit(x)

def rmse(x,y ):
    return np.sqrt( np.nanmean((y -x)**2 ))

colors = ['blue', 'red', 'green']
for i in range(3):
    reg1 = LinearRegression()
    reg1.fit(np.hstack([x for k in range(elm_kwargs['hidden_layer_size'])]), y_logs[:, i])
    xt = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
    preds1 = reg1.predict(np.hstack([xt for k in range(elm_kwargs['hidden_layer_size'])]))
    preds11 = reg1.predict(np.hstack([x for k in range(elm_kwargs['hidden_layer_size'])]))
    plt.scatter(xt, preds1, color=colors[i-1], s=0.2)
    plt.scatter(xt, preds2[:, i], color=colors[i], s=0.8)
    plt.scatter(x, y_logs[:,i], color=colors[i])
    plt.title('ELM: {}; MLR: {}'.format(rmse(preds22[:,i], y_logs[:,i]), rmse(preds11, y_logs[:,i]) ))
    plt.show()

plt.scatter(x, y_logs[:,0], color=colors[0], s=0.1)
plt.scatter(x, y_logs[:,1], color=colors[1], s=0.1)
plt.scatter(x, y_logs[:,2], color=colors[2], s=0.1)

plt.scatter(xt, preds2[:, 0], color=colors[0], s=0.8)
plt.scatter(xt, preds2[:, 1], color=colors[1], s=0.8)
plt.scatter(xt, preds2[:, 2], color=colors[2], s=0.8)
plt.title('BN: {}; NN: {}; AN: {}'.format(*[rmse(preds22[:,i], y_logs[:,i]) for i in range(3)]))
plt.show()
