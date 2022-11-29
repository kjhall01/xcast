from ..core.utilities import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
from operator import sub

def view_reliability(X, Y, x_lat_dim=None, x_lon_dim=None, x_feature_dim=None, x_sample_dim=None, y_lat_dim=None, y_lon_dim=None, y_feature_dim=None, y_sample_dim=None, **kwargs):
    """X = predictions, Y = one hot encoded obs"""

    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    check_all(Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    assert Y.shape[list(Y.dims).index(y_feature_dim)] == X.shape[list(X.dims).index(x_feature_dim)], "Predictions and One-Hot Encoded Observations must have the same number of features"

    x_data = X.stack(point=(x_lat_dim, x_lon_dim, x_sample_dim)).transpose('point', x_feature_dim)
    y_data = Y.stack(point=(y_lat_dim, y_lon_dim, y_sample_dim)).transpose('point', y_feature_dim)


    fig, ax = plt.subplots(nrows=1, ncols=Y.shape[list(Y.dims).index(y_feature_dim)], figsize=(14,4) )
    for i, label in enumerate(Y.coords[y_feature_dim].values):
        xx = x_data.isel(**{x_feature_dim:i }).values.flatten()
        yy = y_data.isel(**{y_feature_dim:i }).values.flatten()
        reliability_diagram(xx, yy, ax=ax[i], title=label, **kwargs)




def reliability_diagram(ypred, t, title=None, tercile_skill_area=True,  perfect_reliability_line=True,   plot_hist=True, fig=None, ax=None, bin_minimum_pct=0.01, scores=True):
    countnonnan = np.ones_like(ypred.squeeze())[~np.isnan(ypred.squeeze())].sum()
    #assert len(ypred.shape) == 2 and ypred.shape[1] == 1, 'ypred must be of shape n_samples x 1'
    ypred = ypred * 0.9999999999999
    assert ypred.shape == t.shape, 'inconsistent shapes between ypred and t - {} vs {}'.format(ypred.shape, t.shape)
    epoelm_rel = []
    epoelm_hist = []
    base_rate = np.nanmean(t)
    uncertainty = base_rate * (1 - base_rate)
    bin_base_rate_diffs = []
    rel_score = []
    for i in range(10):
        forecasts_in_bin = ypred.squeeze()[np.where((ypred.squeeze() >= (i /10.0)) & (ypred.squeeze() < (i/10.0 +0.1)) ) ]
        obs_in_bin = t.squeeze()[np.where((ypred.squeeze() >= (i /10.0)) & (ypred.squeeze() < (i/10.0 +0.1)) ) ]

        #n = forecasts_in_bin.shape[0]
        n = np.ones_like(ypred.squeeze())[np.where((ypred.squeeze() >= (i /10.0)) & (ypred.squeeze() < (i/10.0 +0.1)) ) ].sum()
        m = np.ones_like(ypred.squeeze())[np.where((ypred.squeeze() >= (i /10.0)) & (ypred.squeeze() < (i/10.0 +0.1)) & (t.squeeze() == 1) )].sum()
        epoelm_hist.append(n)
        if n == 0:
            reliability = np.nan
            bin_base_rate = np.nan
            rel=np.nan
        else:
            avg_forecast = np.nanmean(forecasts_in_bin)
            bin_base_rate = np.nanmean(obs_in_bin)
            rel = (avg_forecast - bin_base_rate)**2 * n
            reliability = m / n

        epoelm_rel.append(reliability) # / epoelm_xval.shape[0])
        bin_base_rate_diffs.append((base_rate - bin_base_rate)**2 * n)
        rel_score.append(rel)

    reliability = np.asarray(rel_score)
    reliability = np.nansum(reliability) / t.shape[0]

    bin_base_rate_diffs = np.asarray(bin_base_rate_diffs)
    resolution = np.nansum(bin_base_rate_diffs) / t.shape[0]

    epoelm_rel = np.asarray(epoelm_rel)
    epoelm_hist = np.asarray(epoelm_hist) / countnonnan
    nan = np.where(epoelm_hist < bin_minimum_pct)

    epoelm_hist[nan] = np.nan
    epoelm_rel[nan] = np.nan


    if ax is None:
        ax = plt.gca()

    #plt.hist(epoelm_xval[:, 0], bins=11)
    if tercile_skill_area:
        ur = Polygon([[0.33, 0.33 ], [0.33, 1], [1,1], [1, 1.33/2.0]], facecolor='gray', alpha=0.25)
        bl = Polygon([[0.33, 0.33 ], [0.33, 0], [0,0], [0, 0.33/2.0]], facecolor='gray', alpha=0.25)
        ax.add_patch(ur)
        ax.add_patch(bl)

        ax.text(0.66, 0.28, 'No Resolution')
        noresolution = ax.plot([0, 1], [0.33,0.33], lw=0.5, linestyle='dotted')

        noskill = ax.plot([0, 1], [0.33/2.0,1.33/2.0], lw=0.5, linestyle='dotted')
        figW, figH = ax.get_figure().get_size_inches()
        _, _, w, h = ax.get_position().bounds
        disp_ratio = (figH * h) / (figW * w)
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        angle = (180.0/np.pi)*np.arctan(disp_ratio / data_ratio)
        ax.text(0.66, 0.45, 'No Skill', rotation=angle*0.5)

        ax.plot([0.33, 0.33], [0,1], lw=0.5, linestyle='dotted')


    if perfect_reliability_line:
        figW, figH = ax.get_figure().get_size_inches()
        _, _, w, h = ax.get_position().bounds
        disp_ratio = (figH * h) / (figW * w)
        data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
        angle = (180.0/np.pi)*np.arctan(disp_ratio / data_ratio)
        ax.plot([0, 1], [0,1], lw=0.25, linestyle='dotted')
        ax.text(0.66, 0.58, 'Perfect Reliability', rotation=angle)

    if scores:
        bs = np.nanmean((ypred - t)**2)
        br = np.nanmean((base_rate - t)**2)
        ax.text( 0.7, 0.11, 'BSS: {:0.04f}'.format(1 - (bs/br)))
        ax.text( 0.7, 0.06, 'REL: {:0.04f}'.format(reliability))
        ax.text( 0.7, 0.01, 'RES: {:0.04f}'.format(resolution))
        #ax.text( 0.75, 0.01, 'UNC: {:0.04f}'.format(uncertainty*100))



    ax.set_xlabel('Forecast Probability')
    ax.set_ylabel('Observed Relative Frequency')
    ax.plot([i / 10 +0.05 for i in range(10) ], epoelm_rel, marker='.', lw=1, color='red')

    if plot_hist:
        ax.bar([i / 10 + 0.05 for i in range(10)], epoelm_hist, fill=False, width=0.08, alpha=0.66)

    if title is not None:
        ax.set_title(title)

    return ax
