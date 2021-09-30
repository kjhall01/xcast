from .classification import *
from .core import *
from .flat_estimators import *
from .mme import *
from .preprocessing import *
from .regression import *
from .validation import *
from .verification import *

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

if Path('.xcast_worker_space').is_dir():
	rmrf(Path('.xcast_worker_space'))
Path('.xcast_worker_space').mkdir(exist_ok=True, parents=True)



pointwise_regressors = [
	MultipleLinearRegression,
	PoissonRegression,
	GammaRegression,
	MultiLayerPerceptronRegression,
	RandomForestRegression,
	RidgeRegression,
	ExtremeLearningMachineRegression
]

deterministic_mmes = [
	EnsembleMeanMME,
	BiasCorrectedEnsembleMeanMME,
	MultipleRegressionMME,
	PoissonRegressionMME,
	GammaRegressionMME,
	PrincipalComponentsRegressionMME,
	MultiLayerPerceptronMME,
	RandomForestMME,
	RidgeRegressorMME,
	ExtremeLearningMachineMME,
	ElmPcaMME
]


pointwise_classifiers = [
	ExtendedPOELMClassifier,
	ExtendedMLPClassifier,
	ExtendedNaiveBayesClassifier,
	ExtendedRandomForestClassifier,
	MultivarELRClassifier,
	ExtendedLogisticRegressionClassifier,
	MultiClassPOELMClassifier,
	MultiClassMLPClassifier,
	MultiClassNaiveBayesClassifier,
	MultiClassRandomForestClassifier
]

probabilistic_mmes = [
	MemberCountMME,
	StandardMemberCountMME,
	PoelmMME,
	PoelmPcaMME,
	ProbabilisticMlpMME,
	ProbabilisticRandomForestMME,
	ElrMME,
	MultivariateElrMME
]
