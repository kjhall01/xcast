from ..core.utilities import *
import numpy as np
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def view_roc(X, Y, x_lat_dim=None, x_lon_dim=None, x_feature_dim=None, x_sample_dim=None, y_lat_dim=None, y_lon_dim=None, y_feature_dim=None, y_sample_dim=None ):
    """where X is predicted, and Y is observed"""
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    x_data = X.stack(point=(x_lat_dim, x_lon_dim, x_sample_dim)).transpose('point', x_feature_dim).values
    y_data = Y.stack(point=(y_lat_dim, y_lon_dim, y_sample_dim)).transpose('point', y_feature_dim).values


    tst = x_data *y_data
    x_data = x_data[~np.isnan(tst).any(axis=1)]
    y_data = y_data[~np.isnan(tst).any(axis=1)]
    n_classes = len(X.coords[x_feature_dim].values)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_data[:, i], x_data[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    # Plot all ROC curves
    plt.figure()
    #plt.plot(fpr["macro"], tpr["macro"],
                #label='macro-average ROC curve (area = {0:0.2f})'
                #''.format(roc_auc["macro"]),
                #color='navy', linestyle=':', linewidth=4)
    names=['Below Normal','Near Normal', 'Above Normal']
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label='ROC curve of {0} (AUC = {1:0.2f})'
                ''.format(names[i], roc_auc[i]))

    #plt.plot([0, 1], [0, 1], 'g--', lw=2)#'0.8'
    #plot(x, y, color='green', linestyle='dashed', marker='o',
     #markerfacecolor='blue', markersize=12).
    plt.plot([0, 1], [0, 1], color='0.8', linestyle='dashed', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
