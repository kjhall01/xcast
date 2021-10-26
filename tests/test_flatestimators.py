import pytest
from sklearn.datasets import make_regression, make_classification
import src as xc
import numpy as np

@pytest.mark.parametrize("regressor,x,y", [(reg, *make_regression()) for reg in xc.flat_regressors])
def test_flat_regressor(regressor, x, y):
	reg = regressor()
	reg.fit(x, y.reshape(-1,1))
	preds = reg.predict(x)

@pytest.mark.parametrize('classifier,x,y', [(clf, *make_classification(n_classes=3, n_informative=5)) for clf in xc.flat_classifiers])
def test_flat_classifier(classifier, x, y):
	clf = classifier()
	y1 = np.zeros((y.shape[0], np.max(y)+1))
	for i in range(y1.shape[0]):
		y1[i, y[i]] = 1
	clf.fit(x, y1)
	preds = clf.predict(x)
	probs = clf.predict_proba(x)

def make_elm_params():
	elm_activations = ['sigm', 'tanh', 'relu', 'lin'] #, 'rbf_l1', 'rbf_l2', 'rbf_linf']
	elm_hidden_layer_sizes = [5, 100]
	elm_initializations = ['random', 'pca']
	elm_pruning = ['prune', 'pca', 'none']
	elm_pca = [-999, 0.5]
	elm_c = [ -5, 1, 5]
	elm_preprocessing = ['std', 'minmax', 'none']
	elm_dropconnect = [-1.0, 0.5, 0.9]
	elm_dropout = [-1.0, 0.5, 0.9]
	elm_verbose = [True, False]
	elm_params = []
	for i in elm_activations:
		for j in elm_hidden_layer_sizes:
			for k in elm_initializations:
				for l in elm_pruning:
					for m in elm_pca:
						for n in elm_c:
							for o in elm_preprocessing:
								for p in elm_dropconnect:
									for q in elm_dropout:
										for r in elm_verbose:
											elm_params.append((i, j, k, l ,m, n, o, p, q, r))
	return elm_params

@pytest.mark.slow
@pytest.mark.parametrize('activation,hidden_layer_size,initialization,pruning,pca,c,preprocessing,dropconnect_pr,dropout_pr,verbose', make_elm_params())
def test_elm_regressor(activation,hidden_layer_size,initialization,pruning,pca,c,preprocessing,dropconnect_pr,dropout_pr,verbose):
	x, y = make_regression()
	elm = xc.ELMRegressor(activation,hidden_layer_size,initialization,pruning,pca,c,preprocessing,dropconnect_pr,dropout_pr,verbose)
	elm.fit(x, y.reshape(-1,1))
	preds = elm.predict(x)


def make_poelm_params():
	elm_hidden_layer_sizes = [i+5 for i in range(0, 1000, 300)]
	elm_initializations = ['random', 'pca']
	elm_pruning = ['prune', 'pca', 'none']
	elm_pca = [-999, 0.5, 0.9, 0.999]
	elm_c = [ -5, -1, 0, 1, 5]
	elm_preprocessing = ['std', 'minmax', 'none']
	elm_dropconnect = [-1.0, 0.5, 0.9]
	elm_dropout = [-1.0, 0.5, 0.9]
	elm_verbose = [True, False]
	elm_params = []
	for j in elm_hidden_layer_sizes:
		for k in elm_initializations:
			for l in elm_pruning:
				for m in elm_pca:
					for n in elm_c:
						for o in elm_preprocessing:
							for p in elm_dropconnect:
								for q in elm_dropout:
									for r in elm_verbose:
										elm_params.append((j, k, l ,m, n, o, p, q, r))
	return elm_params

@pytest.mark.slow
@pytest.mark.parametrize('hidden_layer_size,initialization,pruning,pca,c,preprocessing,dropconnect_pr,dropout_pr,verbose', make_poelm_params())
def test_elm_classifier(hidden_layer_size,initialization,pruning,pca,c,preprocessing,dropconnect_pr,dropout_pr,verbose):
	x, y =  make_classification(n_classes=3, n_informative=5)
	elm = xc.POELMClassifier(hidden_layer_size,initialization,pruning,pca,c,preprocessing,dropconnect_pr,dropout_pr,verbose)
	y1 = np.zeros((y.shape[0], np.max(y)+1))
	for i in range(y1.shape[0]):
		y1[i, y[i]] = 1
	elm.fit(x, y1)
	preds = elm.predict(x)
	probs = elm.predict_proba(x)
