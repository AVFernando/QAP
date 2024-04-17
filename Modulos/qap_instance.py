import numpy as np
import random
from copy import deepcopy
from enum import Enum

class QAP_Instance:
    def __init__(self, flow, distance):
        self.flow_matrix = flow
        self.distance_matrix = distance