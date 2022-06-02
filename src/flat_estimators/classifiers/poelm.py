from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from scipy.special import softmax
import datetime as dt
import numpy as np
import scipy.linalg.lapack as la
from scipy.special import softmax
from scipy.spatial.distance import cdist


class POELMClassifier:
    """Probabilistic Output Extreme Learning Machine"""

    def __init__(self, activation='sigm', hidden_layer_size=5, initialization='random', pruning='none', pca=-999, c=1, preprocessing='none', dropconnect_pr=-1.0, dropout_pr=-1.0, verbose=False, threshold=0.5):
        assert type(hidden_layer_size) == int and hidden_layer_size > 0, 'Invalid hidden_layer_size {}'.format(
            hidden_layer_size)
        assert type(initialization) == str and initialization in [
            'random', 'pca'], 'Invalid initialization {}'.format(initialization)
        assert type(pruning) == str and pruning in [
            "none", "prune", "pca"], 'Invalid pruning {}'.format(pruning)
        assert type(pca) in [int, float], 'Invalid pca {}'.format(pca)
        assert type(c) is int, 'Invalid C {}'.format(c)
        assert type(preprocessing) is str and preprocessing in [
            'std', 'minmax', 'none'], 'Invalid preprocessing {}'.format(preprocessing)
        assert type(dropconnect_pr) is float, 'Invalid DropConnect Probability Threshold {}'.format(
            dropconnect_pr)
        assert type(dropout_pr) is float, 'Invalid DropOut Probability Threshold {}'.format(
            dropout_pr)
        assert activation in ['sigm', 'relu', 'lin', 'tanh', 'rbf_l1', 'rbf_l2',
                              'rbf_linf'], 'invalid activation function for poelm- must be one of {}'.format(['sigm', 'relu', 'lin', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf'])

        self.activation = activation
        self.threshold = threshold
        self.initialization = initialization
        self.pruning = pruning
        self.verbose = verbose
        self.dropconnect_pr = dropconnect_pr
        self.dropout_pr = dropout_pr
        self.pca_retained = pca if pca != -1 else None
        self.c = c
        self.hidden_layer_size = hidden_layer_size
        self.preprocessing = preprocessing

    def fit(self, x, y):
        x, y = x.astype(np.float64), y.astype(np.float64)
        assert len(y.shape) == 2, 'POELM expects Y to be of shape (n_samples, n_classes) even if it is a binary classification - got {}'.format(y.shape)
        assert len(
            x.shape) == 2, 'POELM expects X to be of shape (n_samples, n_features) - got {}'.format(x.shape)
        assert 0.0 <= np.min(y) and np.max(
            y) <= 1.0, 'POELM expects that no value in Y is ever greater than 1 or less than 0'
        # if np.mean(np.sum(y, axis=1)) == 1 else False #### Fore now hard-code to true
        self.using_multiclass = True

        y[y < 0.5] = 0.0001
        y[y > 0.5] = 0.9999

        # first, take care of preprocessing
        if self.preprocessing == 'std':
            self.mean, self.std = x.mean(axis=0), x.std(axis=0)
            x = (x - self.mean) / self.std  # scales to std normal dist
            if self.verbose:
                print('{} Applied Standard Normal Scaling '.format(
                    dt.datetime.now()))
        if self.preprocessing == 'minmax':
            self.min, self.max = x.min(axis=0), x.max(axis=0)
            x = ((x - self.min) / (self.max - self.min)) * \
                2 - 1  # scales to [-1, 1]
            if self.verbose:
                print('{} Applied MinMax Scaling '.format(dt.datetime.now()))

        # now, if anything needs to train a PCA on the input data, do it (PCA initialization, or PCA transformation)
        if self.pca_retained != -999 or self.initialization == 'pca':
            self.pca = PCA(n_components=None if self.pca_retained == -
                           999 or self.pca_retained == -1 else self.pca_retained)
            self.pca.fit(x)
            if self.verbose:
                print('{} Fit PCA on X with n_compnents={} '.format(dt.datetime.now(
                ), None if self.pca_retained == -999 or self.pca_retained == -1 else self.pca_retained))

        # apply PCA transformation to input data if applicable
        if self.pca_retained != -999:
            x = self.pca.transform(x)
            if self.verbose:
                print('{} Applied PCA Transformation to X '.format(
                    dt.datetime.now()))

        # after transformation, check feature dim
        x_features, y_features = x.shape[1], y.shape[1]

        # now, initialize weights
        w = np.random.randn(x_features, self.hidden_layer_size)
        b = np.random.randn(1, self.hidden_layer_size)
        if self.verbose:
            print('{} Randomly Initialized W and B '.format(dt.datetime.now()))

        if self.initialization == 'pca':
            w = self.pca.components_[:x_features, :self.hidden_layer_size]
            if self.verbose:
                print('{} Initialized W and B according to PCA coefficients '.format(
                    dt.datetime.now()))

        # now apply dropconnect if applicable
        if self.dropconnect_pr > 1:
            weight_mask = np.random.rand(*w.shape)
            pctl = int(self.dropconnect_pr)
            pctl = np.percentile(w, pctl)
            weight_mask[w >= pctl] = 1.0  # if its greater than pctl, keep it
            # if its less than pctl and less than dropout pr, set to 0
            weight_mask[weight_mask < self.dropconnect_pr / 100.0] = 0.0
            weight_mask[weight_mask > 0] = 1.0  # else set to  1.0
            w = w*weight_mask
            if self.verbose:
                print('{} Applied Biased DropConnect with percentile {} and probability {} '.format(
                    dt.datetime.now(), pctl, self.dropconnect_pr / 100.0))
        elif self.dropconnect_pr > 0:
            weight_mask = np.random.rand(*w.shape)
            weight_mask[weight_mask < self.dropconnect_pr] = 0.0
            weight_mask[weight_mask >= self.dropconnect_pr] = 1.0
            w = w*weight_mask
            if self.verbose:
                print('{} Applied DropConnect with probability {} '.format(
                    dt.datetime.now(), self.dropconnect_pr))

        # now apply dropout if applicable
        if self.dropout_pr > 0:
            self.hidden_neurons = [(w[:, i], b[:, i])
                                   for i in range(w.shape[1])]
            h = np.asarray([self._activate(neuron[0], x, neuron[1])
                           for neuron in self.hidden_neurons]).T
            if self.dropout_pr > 1:
                neuron_mask = np.random.rand(w.shape[1])
                pctl = int(self.dropout_pr)
                pctl = np.percentile(h, pctl)
                neuron_mask[h.mean(axis=0) >= pctl] = 1.0
                neuron_mask[neuron_mask < self.dropout_pr / 100.0] = 0.0
                neuron_mask[neuron_mask > 0] = 1.0
                if np.sum(neuron_mask) < 1:
                    neuron_mask[0] = 1.0
                if self.verbose:
                    print('{} Applied Biased DropOut with percentile {} and probability {} '.format(
                        dt.datetime.now(), pctl, self.dropout_pr / 100.0))
            else:
                neuron_mask = np.random.rand(w.shape[1])
                neuron_mask[neuron_mask < self.dropout_pr] = 0.0
                neuron_mask[neuron_mask >= self.dropout_pr] = 1.0
                if np.sum(neuron_mask) < 1:
                    neuron_mask[0] = 1.0
                if self.verbose:
                    print('{} Applied DropOut with probability {} '.format(
                        dt.datetime.now(), self.dropout_pr))
            w = np.hstack([w[:, i].reshape(-1, 1)
                          for i in range(w.shape[1]) if np.sum(neuron_mask[i]) > 0])

        # now, appy pruning as relevant:
        if self.pruning == 'pca':
            w = PCA(n_components=None if self.pca_retained == -
                    1 or self.pca_retained == -999 else self.pca_retained).fit_transform(w)
            if self.verbose:
                print('{} Applied PCA pruning with n_components of  {} '.format(dt.datetime.now(
                ),  None if self.pca_retained == -1 or self.pca_retained == -999 else self.pca_retained))

        elif self.pruning == "prune":
            self.hidden_neurons = [(w[:, i], b[:, i])
                                   for i in range(w.shape[1])]
            h = np.asarray([self._activate(neuron[0], x, neuron[1])
                           for neuron in self.hidden_neurons]).T

            scores = np.asarray([np.squeeze(chi2(h[:, i].argsort().argsort(
            ).reshape(-1, 1), np.argmax(y, axis=-1)))[0] for i in range(h.shape[1])])
            new_h = []
            for i in range(scores.shape[0]):
                new_h.append(self.hidden_neurons[np.argmax(scores)])
                scores[np.argmax(scores)] = -1

            aics = []
            for i in range(len(scores)):
                self.hidden_neurons = new_h[:i+1]
                h = np.asarray([self._activate(neuron[0], x, neuron[1])
                               for neuron in self.hidden_neurons]).T
                hth = np.dot(np.transpose(h), h)
                hth_plus_ic = hth + np.eye(hth.shape[0]) / (2**self.c)
                ht_logs = np.dot(np.transpose(h), np.log(1 - y) - np.log(y))

                if x.dtype == np.float64 and y.dtype == np.float64:
                    _, B, info = la.dposv(hth_plus_ic, ht_logs)
                elif x.dtype == np.float32 and y.dtype == np.float32:
                    _, B, info = la.sposv(hth_plus_ic, ht_logs)
                else:
                    assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
                        x.dtype, y.dtype)
                if info > 0:
                    hth_plus_ic = hth_plus_ic + np.triu(hth_plus_ic, k=1).T
                    B = np.linalg.lstsq(hth_plus_ic, ht_logs)[0]

                self.beta = -1 * B  # np.dot(np.linalg.pinv(hh), ht)
                preds = self.predict(x, preprocessing=None)
                acc = accuracy_score(np.argmax(y, axis=-1), preds)
                aics.append(self._aic(x.shape[0], acc, i+1))

            aics = np.asarray(aics)
            best = np.argmin(aics)
            self.hidden_neurons = new_h[:best+1]
            w = np.hstack([np.squeeze(neuron[0]).reshape(-1, 1)
                          for neuron in self.hidden_neurons])
            b = np.hstack([np.squeeze(neuron[1]).reshape(-1, 1)
                          for neuron in self.hidden_neurons])
            if self.verbose:
                print('{} Applied AIC / Chi2 Pruning '.format(dt.datetime.now()))
        b = b[:, :w.shape[1]]

        self.hidden_neurons = [(w[:, i], b[:, i]) for i in range(w.shape[1])]
        h = np.asarray([self._activate(neuron[0], x, neuron[1])
                       for neuron in self.hidden_neurons]).T
        hth = np.dot(np.transpose(h), h)
        hth_plus_ic = hth + np.eye(hth.shape[0]) / (2**self.c)
        ht_logs = np.dot(np.transpose(h), np.log(1 - y) - np.log(y))

        if x.dtype == np.float64 and y.dtype == np.float64:
            _, B, info = la.dposv(hth_plus_ic, ht_logs)
        elif x.dtype == np.float32 and y.dtype == np.float32:
            _, B, info = la.sposv(hth_plus_ic, ht_logs)
        else:
            assert False, 'x: {} and y: {} not matching or good dtypes for lapack'.format(
                x.dtype, y.dtype)
        if info > 0:
            hth_plus_ic = hth_plus_ic + np.triu(hth_plus_ic, k=1).T
            B = np.linalg.lstsq(hth_plus_ic, ht_logs)[0]
        self.beta = -1 * B  # np.dot(np.linalg.pinv(hh), ht)
        if self.verbose:
            print('{} Solved POELM '.format(dt.datetime.now()))

    def predict(self, x, preprocessing='asis'):
        x = x.astype(np.float64)
        # first, take care of preprocessing
        ret = self.predict_proba(x, preprocessing=preprocessing)
        if self.using_multiclass:
            return np.argmax(ret, axis=-1)
        else:
            ret[ret < self.threshold] = 0
            ret[ret >= self.threshold] = 1
            return ret

    def predict_proba(self, x, preprocessing='asis'):
        if self.preprocessing == 'std' and preprocessing == 'asis':
            x = (x - self.mean) / self.std  # scales to std normal dist
            if self.verbose:
                print('{} Applied Standard Normal Scaling '.format(
                    dt.datetime.now()))
        if self.preprocessing == 'minmax' and preprocessing == 'asis':
            x = ((x - self.min) / (self.max - self.min)) * \
                2 - 1  # scales to [-1, 1]
            if self.verbose:
                print('{} Applied MinMax Scaling '.format(dt.datetime.now()))

        if self.pca_retained != -999 and preprocessing == 'asis':
            x = self.pca.transform(x)
            if self.verbose:
                print('{} Applied PCA Transformation to Forecast X  '.format(
                    dt.datetime.now()))

        h = np.asarray([self._activate(neuron[0], x, neuron[1])
                       for neuron in self.hidden_neurons]).T
        ret = 1.0 / (1 + np.exp(-1 * np.dot(h, self.beta)))
        # return ret

        if self.using_multiclass:
            sums = np.sum(ret, axis=1)
            return ret / sums.reshape(-1, 1)

        return ret

    def _activate(self, a, x, b):
        if self.activation == 'sigm':
            return 1.0 / (1 + np.exp(-1*np.dot(x, a) + b))
        elif self.activation == 'tanh':
            return np.tanh(np.dot(x, a) + b)
        elif self.activation == 'relu':
            ret = np.dot(x, a) + b
            ret[ret < 0] = 0
            return ret
        elif self.activation == 'lin':
            return np.dot(x, a) + b
        elif self.activation == 'rbf_l1':
            print('Warning: "rbf_l1" activation is broken')
            return np.exp(-cdist(x, a, "cityblock")**2 / b)
        elif self.activation == 'rbf_l2':
            print('Warning: "rbf_l2" activation is broken')
            return np.exp(-cdist(x, a, "euclidean")**2 / b)
        elif self.activation == 'rbf_linf':
            print('Warning: "rbf_linf" activation is broken')
            return np.exp(-cdist(x, a, "chebyshev")**2 / b)
        else:
            assert False, 'Invalid activation function {}'.format(
                self.activation)

    def _aic(self, N, accuracy, S):
        return 2 * N * np.log(((1 - accuracy) / N)**2 / N) + S
