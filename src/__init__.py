from .core import *
from .flat_estimators import *
from .preprocessing import *
from .validation import *
from .verification import *
from .estimators import *
from .visualization import *

import warnings


__version__ = "0.5.8"
__licence__ = "MIT"
__author__ = "KYLE HALL (hallkjc01@gmail.com)"


from pathlib import Path
import zipfile

path = Path(__file__).parents[1]
newdir = Path(str(path).replace('.egg', ''))
if not newdir.is_dir():
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(newdir)
