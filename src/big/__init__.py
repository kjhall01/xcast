from .verification import *
from .mme import *
from .core import *
from .preprocessing import *
from .downscaling import *
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

if (Path().home() / '.xcast_cache').is_dir():
	rmrf(Path.home() / '.xcast_cache')

if not (Path().home() / '.xcast_cache').is_dir():
	(Path().home() / '.xcast_cache').mkdir(parents=True, exist_ok=True)
