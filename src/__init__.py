from .core import align_chunks, guess_coords, check_all
from .flat_estimators import canonical_correlation_analysis, extreme_learning_machine, epoelm, quantile_regression_forest, multivariate_extended_logistic_regression, extended_logistic_regression
from .preprocessing import reformat, remove_climatology, drymask, regrid, gaussian_smooth, Normal, MinMax, OneHotEncoder, GammaTransformer, percentile, EmpiricalTransformer, RollingOneHotEncoder, RollingMinMax
from .validation import CrossValidator, LeaveOneYearOut
from .verification import metric,  kling_gupta_efficiency,  index_of_agreement, rank_probability_score, brier_score,  generalized_receiver_operating_characteristics_score, logarithm_skill_score,  LSS, BrierScore, RankProbabilityScore, GROCS,  IndexOfAgreement, KlingGuptaEfficiency, Spearman, Pearson
from .estimators import BaseEstimator, ACPAC, CCA, PCR, EOF, MELR, ELR, QRF, MLR, ELM, EPOELM, Ensemble
from .visualization import view_probabilistic, view_reliability, reliability_diagram, view_taylor, view_roc, view

__version__ = "0.6.9"
__licence__ = "MIT"
__author__ = "KYLE HALL (hallkjc01@gmail.com)"
