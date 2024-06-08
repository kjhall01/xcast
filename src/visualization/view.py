import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as colors
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy
from cartopy.feature import NaturalEarthFeature
from ..core.utilities import *

def colormap(clevs,cmapname,begin,end,whiteinmiddle):
    # returns a color map (cmap) and its normalization (norm)
    # input variables:
    #   clevs = contour levels
    #   cmapname = matplotlib color map name
    #   begin = a number 0-1 on where to begin in the color map
    #   end = a number 0-1 on where to end in the color map
    #   whiteinmiddle = for contour levels that straddle zero, 'yes' puts white in the middle and 'no' does not
    import matplotlib.pyplot as plt
    cols = plt.get_cmap(cmapname, len(clevs))(np.linspace(begin,end,len(clevs)+1))
    if len(clevs)%2==0 and whiteinmiddle=='yes':
        cols[int(len(clevs)/2),:]=1
    cmap = ListedColormap(cols[1:-1])
    cmap.set_over(cols[-1])
    cmap.set_under(cols[0])
    norm = BoundaryNorm(boundaries=clevs,ncolors=len(clevs)-1)
    return cmap,norm

def lighten_color(color,amount):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def view(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, title='', cross_dateline=False, coastlines=False, borders=True, ocean=True, label=None, label_loc=(0.01, 0.98), savefig=None, drymask=None, **plt_kwargs):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    assert x_sample_dim is None, 'View requires you to select across sample dim to eliminate that dimension first'
    assert x_feature_dim is None, 'View  requires you to select across featyre dim to eliminate that dimension first'

    #check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    assert x_lat_dim in X.coords.keys(), 'XCast requires a dataset_lat_dim to be a coordinate on X'
    assert x_lon_dim in X.coords.keys(), 'XCast requires a dataset_lon_dim to be a coordinate on X'
    check_type(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    if cross_dateline:
        proj = ccrs.PlateCarree(central_longitude=180)
        X = X.assign_coords({x_lon_dim: [i+360 if i < 0 else i for i in X.coords[x_lon_dim].values]}).sortby(x_lon_dim)
        X = X.assign_coords({x_lon_dim: X.coords[x_lon_dim].values - 180})
    else:
        proj = ccrs.PlateCarree()


    mask = X.where(np.isnan(X), other=1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 9), subplot_kw={'projection': proj})
    CS3 = X.plot(ax=ax, add_colorbar=False, **plt_kwargs)

    if drymask is not None:
        dmcmap = plt.get_cmap('RdBu').copy()
        dmcmap.set_under('lavenderblush')
        drymask = xr.ones_like(drymask).where(np.isnan(drymask), other=np.nan)
        drymask.plot(ax=ax, add_colorbar=False, vmin=22, vmax=23, cmap=dmcmap)

    axins2_bottom = inset_axes(ax, width="100%",  height="5%", loc='lower center', bbox_to_anchor=(-0.0, -0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0.1 )

    cbar_fbc = fig.colorbar(CS3, ax=ax,  cax=axins2_bottom, orientation='horizontal')
    cbar_fbc.set_label(X.name)

    if ocean:
        ocean = NaturalEarthFeature(category='physical',name='ocean',scale='50m')
        ax.add_feature(ocean,facecolor= lighten_color('lightblue',0.3),edgecolor='none')

    if coastlines:
        ax.coastlines()

    if borders is True:
        countryshp = shpreader.natural_earth(resolution='50m',category='cultural',name='admin_0_countries')
        countryreader = shpreader.Reader(countryshp)
        countries = countryreader.records()
        for country in countries:
            ax.add_geometries([country.geometry],ccrs.PlateCarree(),facecolor='none',edgecolor='black',lw=0.75)
    elif borders:
        countryreader = shpreader.Reader(borders)
        countries = countryreader.records()
        for country in countries:
            ax.add_geometries([country.geometry],ccrs.PlateCarree(),facecolor='none',edgecolor='black',lw=0.75)
    else:
        pass

    if label is not None:
        props = dict( facecolor='white', alpha=1, edgecolor='black')
        ax.text(label_loc[0], label_loc[1], label, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=props)


    ax.set_title(title)

    if savefig is not None:
        plt.savefig(savefig, dpi=100)
    return ax