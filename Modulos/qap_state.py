import numpy as np
import random
from copy import deepcopy
from enum import Enum

class QAP_State:
    def __init__(self, instance, assigned=[0]):
        self.instance = instance
        self.assigned = assigned
        self.not_assigned = list(set(range(len(instance.flow_matrix))) - set(self.assigned))
        self.all_assigned = len(self.not_assigned) == 0
        self.update_cost()

    def calculate_cost(self, i, j):
        place1, place2 = self.assigned[i], self.assigned[j]
        flow_cost = self.instance.flow_matrix[place1, place2]
        distance_cost = self.instance.distance_matrix[i, j]
        return flow_cost * distance_cost

    def update_cost(self):
        if len(self.assigned) > 1:
            self.cost = sum(self.calculate_cost(i, j) for i in range(len(self.assigned)) for j in range(len(self.assigned)) if i != j)
        else:
            self.cost = 0

    def __deepcopy__(self, memo):
        return type(self)(instance=self.instance, assigned=deepcopy(self.assigned))

    def __str__(self):
        return f"Asignacion en la lista: {self.assigned}\nCosto de la asignacion: {self.cost}"
