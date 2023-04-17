import numpy as np 
from ..flat_estimators.einstein_epoelm import epoelm 
from ..verification.flat_metrics import kling_gupta_efficiency 
import itertools
import datetime as dt
from sklearn.model_selection import KFold

def score_decorator(func):
    def func1(params, x, y, estimator=epoelm):
        indices = np.arange(x.shape[0])
        ndx = 1
        xval_y, xval_predicted_cdfs = [], []
        kf = KFold(n_splits=5)
        for xtrainndx, xtestndx in kf.split(indices):
            xtrain, ytrain = x[xtrainndx, :], y[xtrainndx, :]
            xtest, ytest = x[xtestndx, :], y[xtestndx, :]
            xval_y.append(ytest)
            epoelm = estimator(**params)
            epoelm.fit(xtrain, ytrain)
            predicted = epoelm.predict(xtest)
            xval_predicted_cdfs.append(predicted)
            ndx += 1

        xval_predicted_cdfs = np.vstack(xval_predicted_cdfs)
        xval_y = np.vstack(xval_y)
        return  func(xval_predicted_cdfs, xval_y) 
    func1.__name__ = func.__name__
    return func1

@score_decorator
def get_score(x, y):
    return kling_gupta_efficiency(x, y)

def sort_queue_by_score(queue, scores):
    scores = np.asarray(scores)
    ranks = np.argsort(scores)
    queue = [ queue[i] for i in ranks]
    scores = [ scores[i] for i in ranks]
    return queue, scores

def get_random_selection(gene_set):
    ret = {}
    for key in gene_set.keys():
        ret[key] = gene_set[key][np.random.randint(len(gene_set[key]))]
    return ret

def DFS(x,y, queue_len=5, gene_set=None, generation_size=5, n_mutations=2, lag=10, tol=0.001, estimator=epoelm, scorer=get_score):
    assert gene_set is not None, 'Must provide "Gene-Set" for hyperparameter tuning'
    #initialize queue
    best_params= get_random_selection(gene_set)
    best_score = scorer(best_params, x, y, estimator=estimator )
    queue = [ get_random_selection(gene_set) for j in range(queue_len-1) ]
    scores = [ scorer(i, x, y, estimator=estimator) for i in queue]
    queue, scores = sort_queue_by_score(queue, scores)
    history = [-999 + i for i in range(lag)]
    count = 3
    while len(queue) > 0 and  np.abs(best_score - history[-(lag-1)]) > tol:
        #print("    N Queued: {:>02.2f}, BestScore: {:1.7f}, WorstQueued: {:1.7}, TotalTested: {:>04}".format(len(queue), best_score, scores[0], count), end='\n')
        current = queue.pop(-1)
        current_score = scores.pop(-1)
        if current_score > best_score and ~np.isnan(current_score) and current_score < 1:
            best_score = current_score
            best_params = current
        for i in range(generation_size):
            params = current.copy()
            kys = [key for key in params.keys()]
            for j in range(n_mutations):
                ky = kys[np.random.randint(len(kys))]
                params[ky] = gene_set[ky][np.random.randint(len(gene_set[ky]))]
            score = scorer(params, x, y, estimator)
            count += 1
            makes_it = False
            for s in scores:
                if score > s:
                    makes_it = True
            if makes_it:
                scores.append(score)
                queue.append(params)
            queue, scores = sort_queue_by_score(queue, scores)
            history.append(scores[0] if len(scores) > 0 else -999)
            n = history.pop(0)

        # prune queue
        while len(queue) > queue_len:
            queue.pop(0)
            scores.pop(0)
    return  best_params, best_score, 0