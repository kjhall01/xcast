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

Path('.xcast_worker_space').mkdir(exist_ok=True, parents=True)
__version__ = "0.2.6"

def clear_cache():
	if Path('.xcast_worker_space').is_dir():
		rmrf(Path('.xcast_worker_space'))
		Path('.xcast_worker_space').mkdir(exist_ok=True, parents=True)
