from .deterministic import *
from .probabilistic import *

deterministic_mmes = [
	mEnsembleMean,
	mBiasCorrectedEnsembleMean,
	mMultipleRegression,
	mPoissonRegression,
	mGammaRegression,
	mPrincipalComponentsRegression,
	mMultiLayerPerceptron,
	mRandomForest,
	mRidgeRegressor,
	mExtremeLearningMachine,
	mExtremeLearningMachinePCA
]

probabilistic_mmes = [
	pMemberCount,
	pStandardMemberCount,
	pExtendedLogisticRegression,
	pMultivariateELR,
	pRandomForest,
	pMultiLayerPerceptron,
	pPCAPOELM,
	pPOELM
]
