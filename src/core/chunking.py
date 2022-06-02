from .utilities import guess_coords, check_all, shape


def align_chunks(X, Y, lat_chunks=5, lon_chunks=5, x_lat_dim=None, x_lon_dim=None, y_lat_dim=None, y_lon_dim=None, x_feature_dim=None, y_feature_dim=None, x_sample_dim=None, y_sample_dim=None):
    x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim = guess_coords(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim = guess_coords(
        Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    check_all(X, x_lat_dim, x_lon_dim,  x_sample_dim, x_feature_dim)
    check_all(Y, y_lat_dim, y_lon_dim,  y_sample_dim, y_feature_dim)

    x_lat_shape, x_lon_shape, x_samp_shape, x_feat_shape = shape(
        X, x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    y_lat_shape, y_lon_shape, y_samp_shape, y_feat_shape = shape(
        Y, y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    X1 = X.chunk({x_lat_dim: max(x_lat_shape // lat_chunks, 1),
                 x_lon_dim: max(x_lon_shape // lon_chunks, 1)})
    Y1 = Y.chunk({y_lat_dim: max(y_lat_shape // lat_chunks, 1),
                 y_lon_dim: max(y_lon_shape // lon_chunks, 1)})

    X1 = X1.transpose(x_lat_dim, x_lon_dim, x_sample_dim, x_feature_dim)
    Y1 = Y1.transpose(y_lat_dim, y_lon_dim, y_sample_dim, y_feature_dim)

    return X1, Y1
