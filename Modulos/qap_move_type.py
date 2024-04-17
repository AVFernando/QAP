import numpy as np
import random
from copy import deepcopy
from enum import Enum

class MoveType(Enum):
    CONSTRUCTIVE = 1
    SWAP = 2
    TWO_OPT = 3
    RELOCATION = 4
