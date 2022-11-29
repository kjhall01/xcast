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


def view_probabilistic(X, x_lat_dim=None, x_lon_dim=None, x_sample_dim=None, x_feature_dim=None, title='', coastlines=False, borders=True, ocean=True, label=None, label_loc=(0.01, 0.98), savefig=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords_view_prob(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    assert x_sample_dim is None, 'View probabilistic requires you to select across sample dim to eliminate that dimension first'
    #check_all(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    assert x_lat_dim in X.coords.keys(), 'XCast requires a dataset_lat_dim to be a coordinate on X'
    assert x_lon_dim in X.coords.keys(), 'XCast requires a dataset_lon_dim to be a coordinate on X'
    assert x_feature_dim in X.coords.keys(), 'XCast requires a dataset_feature_dim to be a coordinate on X'
    check_type(X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)

    bounds = [40,45,50,55,60,65,70,75,80]
    nbounds = [40,45,50]

    bn_cmap = plt.get_cmap('BrBG_r')  # define the colormap
    bn_cmaplist = [bn_cmap(i) for i in range(bn_cmap.N)][bn_cmap.N//2:]

    # create the new map
    bn_cmap = mpl.colors.LinearSegmentedColormap.from_list('BNCMAP', bn_cmaplist, bn_cmap.N//2)
    mpl.colormaps.register(bn_cmap)
    # define the bins and normalize
    bn_norm = mpl.colors.BoundaryNorm(bounds, bn_cmap.N//2)

    an_cmap = plt.get_cmap('BrBG')  # define the colormap
    an_cmaplist = [an_cmap(i) for i in range(an_cmap.N)][an_cmap.N//2:]

    # create the new map
    an_cmap = mpl.colors.LinearSegmentedColormap.from_list('ANCMAP', an_cmaplist, an_cmap.N//2)
    mpl.colormaps.register(an_cmap)
    # define the bins and normalize
    an_norm = mpl.colors.BoundaryNorm(bounds, an_cmap.N//2)


    nn_cmap = plt.get_cmap('Greys')  # define the colormap
    nn_cmaplist = [nn_cmap(i) for i in range(nn_cmap.N)][:nn_cmap.N//2+2]

    # create the new map
    nn_cmap = mpl.colors.LinearSegmentedColormap.from_list('NNCMAP', nn_cmaplist, nn_cmap.N//2)
    mpl.colormaps.register(nn_cmap)
    # define the bins and normalize
    nn_norm = mpl.colors.BoundaryNorm(nbounds, nn_cmap.N//2)


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 9), subplot_kw={'projection': ccrs.PlateCarree()})
    bounds = [40,45,50,55,60,65,70,75,80]
    nbounds = [40,45,50]
    mask = X.mean(x_feature_dim)
    mask = mask.where(np.isnan(mask), other=1)
    argmax = X.fillna(-999).argmax(x_feature_dim) * mask

    flat = mask.where(argmax != 2, other=X.isel(M=2))
    flat = flat.where(argmax != 1, other=X.isel(M=1))
    flat = flat.where(argmax != 0, other=X.isel(M=0)) * mask


    CS3 = flat.where(argmax == 2, other=np.nan).plot(ax=ax, add_colorbar=False, vmin=0.35, vmax=0.85, cmap=plt.get_cmap('ANCMAP', 10))
    CS1 = flat.where(argmax == 0, other=np.nan).plot(ax=ax, add_colorbar=False, vmin=0.35, vmax=0.85, cmap=plt.get_cmap('BNCMAP', 10))
    CS2 = flat.where(argmax == 1, other=np.nan).plot(ax=ax, add_colorbar=False, vmin=0.35, vmax=0.55, cmap=plt.get_cmap('NNCMAP', 4))

    axins_f_bottom = inset_axes(ax, width="35%", height="5%", loc='lower left', bbox_to_anchor=(-0, -0.15, 1, 1), bbox_transform=ax.transAxes,borderpad=0.1 )
    axins2_bottom = inset_axes(ax, width="20%",  height="5%", loc='lower center', bbox_to_anchor=(-0.0, -0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0.1 )
    axins3_bottom = inset_axes(ax, width="35%",  height="5%", loc='lower right', bbox_to_anchor=(0, -0.15, 1, 1), bbox_transform=ax.transAxes, borderpad=0.1 )


    cbar_fbl = fig.colorbar(CS1, ax=ax, cax=axins_f_bottom, orientation='horizontal')
    cbar_fbl.set_label('Below-Normal (%)')
    cbar_fbl.set_ticks([i /100.0 for i in bounds])
    cbar_fbl.set_ticklabels(bounds)


    cbar_fbc = fig.colorbar(CS2, ax=ax,  cax=axins2_bottom, orientation='horizontal')
    cbar_fbc.set_label('Near-Normal (%)')
    cbar_fbc.set_ticks([i /100.0 for i in nbounds])
    cbar_fbc.set_ticklabels(nbounds)

    cbar_fbr = fig.colorbar(CS3, ax=ax,  cax=axins3_bottom, orientation='horizontal')
    cbar_fbr.set_label('Above-Normal (%)')
    cbar_fbr.set_ticks([i /100.0 for i in bounds])
    cbar_fbr.set_ticklabels(bounds)

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
