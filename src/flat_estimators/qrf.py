import numpy as np
from sklearn.ensemble import RandomForestRegressor
from numba import jit, prange, set_num_threads

### this file contains a modified version of PyQuantRF - but with a much less stupid implementation

@jit(nopython=True, parallel=False)
def find_quant(trainy, train_tree_node_ID, pred_tree_node_ID, qntl):
    """find_quant(trainy, train_tree_node_ID, pred_tree_node_ID, qntl)

    Aggregates the leaves from the random forest and calculates the quantiles.
    Aggregates leaves based on the tree node indexes from both the training
    and prediction data. Values from the training target data is then used
    to rebuild the leaves for each prediction, which is then summarized
    to the specified quantiles. This is the slowest step in the process,
    so numba is used to speed up this step.
    Parameters
    ----------
    trainy : numpy array of shape (n_target)
        The origianl training target data
    train_tree_node_ID : numpy array of shape (n_training_samples, n_trees)
        array of leaf indices from the training data
    pred_tree_node_ID : numpy array of shape (n_predict_samples, n_trees)
        array of leaf indices from the prediction data
    qntl : numpy array
        quantiles used, must range from 0 to 1
    Returns
    -------
    out : numpy array of shape (n_predict_samples, n_qntl)
        prediction for each quantile
    """
    npred = pred_tree_node_ID.shape[0]
    out = np.zeros((npred, qntl.size))*np.nan
    for i in prange(pred_tree_node_ID.shape[0]):
        idxs = np.where(train_tree_node_ID == pred_tree_node_ID[i, :])[0]
        sample = trainy[idxs]
        out[i, :] = np.quantile(sample, qntl)
    return out


@jit(nopython=True, parallel=False)
def find_nonexceed(trainy, train_tree_node_ID, pred_tree_node_ID, thres):
    """find_quant(trainy, train_tree_node_ID, pred_tree_node_ID, qntl)

    Aggregates the leaves from the random forest and calculates the quantiles.
    Aggregates leaves based on the tree node indexes from both the training
    and prediction data. Values from the training target data is then used
    to rebuild the leaves for each prediction, which is then summarized
    to the specified quantiles. This is the slowest step in the process,
    so numba is used to speed up this step.
    Parameters
    ----------
    trainy : numpy array of shape (n_target)
        The origianl training target data
    train_tree_node_ID : numpy array of shape (n_training_samples, n_trees)
        array of leaf indices from the training data
    pred_tree_node_ID : numpy array of shape (n_predict_samples, n_trees)
        array of leaf indices from the prediction data
    qntl : numpy array
        quantiles used, must range from 0 to 1
    Returns
    -------
    out : numpy array of shape (n_predict_samples, n_qntl)
        prediction for each quantile
    """
    
    npred = pred_tree_node_ID.shape[0]
    out = np.zeros((npred, thres.shape[0]))*np.nan
    for i in prange(pred_tree_node_ID.shape[0]):
        for j in prange(thres.shape[0]):
            idxs = np.where(train_tree_node_ID == pred_tree_node_ID[i, :])[0]
            sample = trainy[idxs]
            out[i, j] = (sample < thres[j]).sum() / float(sample.shape[0])
    return out



class quantile_regression_forest:
    """A quantile random forest regressor based on the scikit-learn RandomForestRegressor

    A wrapper around the RandomForestRegressor which summarizes based on quantiles rather than
    the mean. Note that quantile predicitons take much longer than mean predictions.
    Parameters
    ----------
    nthreads : int, default=1
        number of threads to used
    rf_kwargs : array or array like
        kwargs to be passed to the RandomForestRegressor

    See Also
    --------
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html?highlight=randomforestregressor#sklearn.ensemble.RandomForestRegressor.apply
    """

    def __init__(self, nthreads=1, **rf_kwargs):
        rf_kwargs['n_jobs'] = nthreads
        self.forest = RandomForestRegressor(**rf_kwargs)
      #  set_num_threads(nthreads)

    def fit(self, X, y, sample_weight=None):
        """
        Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
        """
        self.forest.fit(X, y.ravel(), sample_weight)
        self.trainy = y.copy()
        ntrees = self.forest.n_estimators
        ntrain = y.shape[0]
        self.train_tree_node_ID = np.zeros([ntrain, ntrees])
        for i in range(ntrees):
            self.train_tree_node_ID[:, i] = self.forest.estimators_[i].apply(X)
        self.train_tree_node_ID = self.train_tree_node_ID.astype('h') # because it is only indexes, store as short int

    def predict(self, X, quantile=0.5):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        qntl : {array-like} of shape (n_quantiles)
            Quantile or sequence of quantiles to compute, which must be between
            0 and 1 inclusive. Passed to numpy.quantile.
        Returns
        -------
        y : ndarray of shape (n_samples, n_qntl)
            The predicted values.
        """
        qntl = np.asanyarray(quantile)
        ntrees = self.forest.n_estimators
        ntrain = self.trainy.shape[0]
        npred = X.shape[0]
        pred_tree_node_ID = np.zeros([npred, ntrees])

        for i in range(ntrees):
            pred_tree_node_ID[:, i] = self.forest.estimators_[i].apply(X)

        return find_quant(self.trainy, self.train_tree_node_ID, pred_tree_node_ID.astype('h'), qntl)


    def predict_proba(self, X, quantile=None):
        """
        Predict regression target for X.
        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the trees in the forest.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        qntl : {array-like} of shape (n_quantiles)
            Quantile or sequence of quantiles to compute, which must be between
            0 and 1 inclusive. Passed to numpy.quantile.
        Returns
        -------
        y : ndarray of shape (n_samples, n_qntl)
            The predicted values.
        """
        if quantile is None:
            qntl = [0.33, 0.67]
        else:
            qntl = quantile
        
        qntl = np.asarray(qntl).flatten()
        thres = np.quantile(self.trainy, qntl)
        ntrees = self.forest.n_estimators
        ntrain = self.trainy.shape[0]
        npred = X.shape[0]
        pred_tree_node_ID = np.zeros([npred, ntrees])

        for i in range(ntrees):
            pred_tree_node_ID[:, i] = self.forest.estimators_[i].apply(X)

        pne = find_nonexceed(self.trainy, self.train_tree_node_ID, pred_tree_node_ID.astype('h'), thres)
        if quantile is None:
            bn = pne[:, 0].reshape(-1,1)
            an = 1 - pne[:, 1].reshape(-1,1)
            nn = 1 - bn - an
            return np.hstack([bn, nn, an])
        else:
            return pne


    def apply(self, X):
        """
        wrapper for sklearn.ensemble.RandomForestRegressor.apply
        Apply trees in the forest to X, return leaf indices.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        return self.forest.apply(X)

    def decision_path(self, X):
        """
        wrapper for sklearn.ensemble.RandomForestRegressor.decision_path
        Return the decision path in the forest.
        .. versionadded:: 0.18
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.
        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        return self.forest.decision_path(X)

    def set_params(self, **params):
        """
        wrapper for sklearn.ensemble.RandomForestRegressor.set_params
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : object
            Estimator instance.
        """
        return self.forest.set_params(**params)


