from ..core.utilities import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import stats

def view_taylor(X, Y, x_lat_dim=None, x_lon_dim=None, x_feature_dim=None, x_sample_dim=None, y_lat_dim=None, y_lon_dim=None, y_feature_dim=None, y_sample_dim=None, loc="lower left" ):
    """where X is predicted, and Y is observed"""
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    X1 = X.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    Y1 = Y.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)
    #x_data = X1.values.reshape(len(X1.coords[x_lat_dim].values)*len(X1.coords[x_lon_dim].values)*len(X1.coords[x_sample_dim].values), len(X1.coords[x_feature_dim].values))
    #y_data = Y1.values.reshape(len(Y1.coords[y_lat_dim].values)*len(Y1.coords[y_lon_dim].values)*len(Y1.coords[y_sample_dim].values), len(Y1.coords[y_feature_dim].values))
    x_data = X.stack(point=(x_lat_dim, x_lon_dim, x_sample_dim)).transpose('point', x_feature_dim).values
    y_data = Y.stack(point=(y_lat_dim, y_lon_dim, y_sample_dim)).transpose('point', y_feature_dim).values


    tst = x_data *y_data
    x_data = x_data[~np.isnan(tst).any(axis=1)]
    y_data = y_data[~np.isnan(tst).any(axis=1)]
    n_classes = len(X.coords[x_feature_dim].values)

    obs_stddev = y_data.std()
    stddevs = x_data.std(axis=0)

    correlations = []
    for i in range(n_classes):
        try:
            coef, p = stats.pearsonr(np.squeeze(x_data[:,i]).astype(float), np.squeeze(y_data[:,0]).astype(float))
            correlations.append(coef)
        except:
            correlations.append(np.nan)

    obs_cor = 1.0
    correlations = np.asarray(correlations)
    obs_rmsd = 0
    rmsds = np.sqrt(obs_stddev**2 + stddevs**2 - 2* obs_stddev*stddevs*correlations)

    angles = (1 - correlations ) * np.pi / 2.0

    xs = [np.cos(angles[i]) * stddevs[i] for i in range(stddevs.shape[0])]
    ys = [np.sin(angles[i]) * stddevs[i] for i in range(stddevs.shape[0])]

    fig = plt.figure(frameon=False, figsize=(5,5))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'purple', 'green'])
    for i, color in zip(range(len(xs)), colors):
        plt.scatter(xs[i], ys[i], color=color, lw=2, label='Model {}'.format(X.coords[x_feature_dim].values[i]))
    plt.scatter(obs_stddev, 0, color='red', label='Observations')

    for i in range(4):
        circle1 = plt.Circle((obs_stddev, 0), max(rmsds)*((i+1) / 4.0), edgecolor='green', fill=False, alpha=0.5, linestyle='-.')
        fig.axes[0].add_patch(circle1)
        fig.axes[0].annotate('{:>02.2}'.format(max(rmsds)*((i+1) / 4.0)), (obs_stddev, max(rmsds)*((i+1.1) / 4.0)), (obs_stddev, max(rmsds)*((i+1.1) / 4.0)), color='green', alpha=0.5, size=8)

    fig.axes[0].annotate('RMS', (obs_stddev, max(rmsds)*1.1), (obs_stddev, max(rmsds)*1.1), color='green', alpha=0.5)


    for i in range(7):
        circle1 = plt.Circle((0, 0), obs_stddev*(i / 3.0), edgecolor='black', fill=False)
        fig.axes[0].add_patch(circle1)


    for i in range(5):
        angle = np.pi / 2.0 * (1 - (i+0.5)/5.0)
        plt.plot([0, np.cos(angle)*obs_stddev*1.5], [0, np.sin(angle)*obs_stddev*1.5], linewidth=0.5, color='blue', alpha=0.5, linestyle='-.')
        fig.axes[0].annotate('{}'.format((i+0.5) / 5.0), (np.cos(angle)*obs_stddev*1.5, np.sin(angle)*obs_stddev*1.5), (np.cos(angle)*obs_stddev*1.5, np.sin(angle)*obs_stddev*1.5), alpha=0.5, color='blue', size=8, rotation=(1 - (i+0.5)/5.0)*90)
    fig.axes[0].annotate('Peason Correlation', (np.cos(np.pi/4)*obs_stddev*1.45, np.sin(np.pi/4)*obs_stddev*1.45), (np.cos(np.pi/4)*obs_stddev*1.45, np.sin(np.pi/4)*obs_stddev*1.45), color='blue', rotation=315, alpha=0.5)

    plt.xlim([0,  obs_stddev * 1.6])
    plt.ylim([0,  obs_stddev * 1.6])
    plt.xticks([obs_stddev * i / 3.0 for i in range(5)], ['{:<02.2}'.format(obs_stddev * i / 3.0) for i in range(5)])
    plt.yticks([obs_stddev * i / 3.0 for i in range(5)], ['{:<02.2}'.format(obs_stddev * i / 3.0) for i in range(5)])
    plt.xlabel('Standard Deviation')
    plt.ylabel('Standard Deviation')
    plt.title('Taylor Diagram')
    plt.legend(loc=loc)
