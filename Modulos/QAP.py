import numpy as np
import random
from copy import deepcopy
from enum import Enum

class QAP_Instance:
    def __init__(self, flow, distance):
        self.flow_matrix = flow
        self.distance_matrix = distance
        print("\nMatrices generadas aleatoriamente:")
        print("Matriz de flujo:")
        print(self.flow_matrix)
        print("\nMatriz de distancia:")
        print(self.distance_matrix)

class QAP_State:
    def __init__(self, instance, assigned=[0]):
        self.instance = instance
        self.assigned = assigned
        self.not_assigned = list(set(range(len(instance.flow_matrix))) - set(self.assigned))
        self.all_assigned = len(self.not_assigned) == 0
        self.cost = 0
        self.update_cost()

    def calculate_cost(self, i, j):
        place1 = self.assigned[i]
        place2 = self.assigned[j]

        flow_cost = self.instance.flow_matrix[place1, place2]
        distance_cost = self.instance.distance_matrix[i, j]

        return flow_cost * distance_cost


    def update_cost(self):
        n = len(self.assigned)
        if n > 1:
            cost = sum(self.calculate_cost(i, j) for i in range(n) for j in range(n) if i != j)
            self.cost = cost
        else:
            self.cost = 0


    def __deepcopy__(self, memo):
        copy_instance = type(self)(instance=self.instance, assigned=deepcopy(self.assigned))
        return copy_instance

    def __str__(self):
        return f"Asignacion en la lista: {self.assigned}\nCosto de la asignacion: {self.cost}"

class move_type(Enum):
    CONSTRUCTIVE = 1
    SWAP = 2
    TWO_OPT = 3
    RELOCATION = 4

class QAP_Environment():
    @staticmethod
    def gen_actions(state, move_type, shuffle=False):
        if move_type == move_type.CONSTRUCTIVE:
            actions = [(move_type.CONSTRUCTIVE, place) for place in state.not_assigned]

        elif move_type == move_type.SWAP:
            n_len = len(state.assigned)
            actions = [(move_type.SWAP, (i, j)) for i in range(n_len) for j in range(i + 1, n_len)]

        elif move_type == move_type.TWO_OPT:
            n_len = len(state.assigned)
            actions = [(move_type.TWO_OPT, (i, j)) for i in range(n_len) for j in range(i - 1, i + 2) if j != i]

        elif move_type == move_type.RELOCATION:
            n_len = len(state.assigned)
            actions = []
            for i in range(n_len):
                for j in range(n_len):
                    if i != j:
                        actions.append((move_type, (i, j)))

        else:
            raise NotImplementedError(f"Tipo de accion '{move_type}' no implementado para QAP")

        if shuffle:
            random.shuffle(actions)

        for action in actions:
            yield action

    @staticmethod
    def state_transition(state, action):
        if action[0] == move_type.CONSTRUCTIVE and not state.all_assigned:
            state.assigned.append(action[1])
            state.not_assigned.remove(action[1])

            if len(state.not_assigned) == 0:
                state.update_cost()
                state.all_assigned = True

        elif action[0] == move_type.SWAP and state.all_assigned == True:
            print("\nMomento de hacer swap o two-opt")
            i, j = action[1]
            if i not in state.assigned or j not in state.assigned:
                raise ValueError(f"Las instalaciones {i} y/o {j} no están asignadas.")
            i_idx = state.assigned.index(i)
            j_idx = state.assigned.index(j)
            state.assigned[i_idx] = j
            state.assigned[j_idx] = i
            state.update_cost()

        elif action[0] == move_type.RELOCATION and state.all_assigned == True:
            print("\nMomento de hacer relocation")
            i, j = action[1]
            if i not in state.assigned or j not in state.assigned:
                raise ValueError(f"Las instalaciones {i} y/o {j} no están asignadas.")
            i, j = action[1]
            elem = state.assigned.pop(i)
            state.assigned.insert(j, elem)
            state.update_cost()

        else:
            raise NotImplementedError(f"Tipo de accion '{action[0]}' no implementado para QAP")

        return state

    def calculate_cost_after_action(state, action):
        if action[0] == move_type.SWAP:
            k, l = action[1]

            # Swap the facilities in the assignment array
            state.assignment_array[k], state.assignment_array[l] = \
                state.assignment_array[l], state.assignment_array[k]

            # Calculate the change in cost caused by the swap
            old_cost = state.cost

            # Calculate the change in cost for the swapped facilities
            change_in_cost = 0
            n = len(state.assignment_array)
            for i in range(n):
                if i != k and i != l:  # Exclude the swapped facilities
                    facility1 = state.assignment_array[i]
                    facility2_k = state.assignment_array[k]
                    facility2_l = state.assignment_array[l]

                    location1 = i
                    location2_k = k
                    location2_l = l

                    # Calculate the change in cost for swapping k and i with l and i
                    change_in_cost += (state.instance.flow_matrix[facility1, facility2_l] - 
                                       state.instance.flow_matrix[facility1, facility2_k]) * \
                                        state.instance.distance_matrix[location1, location2_k] + \
                                        (state.instance.flow_matrix[facility1, facility2_k] - 
                                       state.instance.flow_matrix[facility1, facility2_l]) * \
                                        state.instance.distance_matrix[location1, location2_l]

            # Update the cost with the change
            new_cost = old_cost + change_in_cost
            state.cost = new_cost

            # Swap back to the original assignment array
            state.assignment_array[k], state.assignment_array[l] = \
                state.assignment_array[l], state.assignment_array[k]

            return new_cost

        return state.cost

def evalConstructiveActions(state, env):
    evals = []
    n = len(state.assigned)
    for action in env.gen_actions(state, move_type.CONSTRUCTIVE):
        place2 = action[1]
        
        if place2 in state.assigned:
            continue
        cost = 0
        for i, place1 in enumerate(state.assigned):
            cost += state.instance.flow_matrix[place1, place2] * \
                    state.instance.distance_matrix[i, n]
        evals.append((action, -cost))
    return evals

class GreedyAgent:
    def __init__(self, eval_actions):
        self.eval_actions = eval_actions

    def select_action(self, evals):
        return max(evals, key=lambda x: x[1])[0]

    def action_policy(self, state, env):
        evals = self.eval_actions(state, env)
        if len(evals) == 0:
            return None
        return self.select_action(evals)

    def __deepcopy__(self, memo):
        copy_instance = type(self)(eval_actions=self.eval_actions)
        return copy_instance

class LocalSearchAgent(GreedyAgent):
    def __init__(self, action_type, first_improvement=True):
        self.action_type = action_type
        self.first_improvement = first_improvement

    def eval_actions(self, state, env):
        current_cost = state.cost
        evals = []
        
        for action in env.gen_actions(state, self.action_type, shuffle=True):
            new_cost = env.calculate_cost_after_action(state, action)
            if new_cost < current_cost:
                evals.append((action, new_cost))
                if self.first_improvement:
                    return evals
        return evals
    
    def __deepcopy__(self, memo):
        new_instance = type(self) (
            action_type = self.action_type,
            first_improvement = self.first_improvement
        )
        return new_instance

#--------------------------------------------------------------------------------
flow = np.array([[0, 3, 1, 2], 
                 [3, 0, 1, 1], 
                 [1, 1, 0, 4], 
                 [2, 1, 4, 0]])

distance = np.array([[0,    22,   53, 1000],
                     [22,   0,    40, 1000],
                     [53,   40,   0,  55],  
                     [1000, 1000, 55, 0]])  

env = QAP_Environment()
instance = QAP_Instance(flow, distance)
current_state = QAP_State(instance)

agent = GreedyAgent(evalConstructiveActions)
while True:
    action = agent.action_policy(current_state, env)
    if action is None:
        break
    current_state = env.state_transition(current_state, action)
print(f"\nEstado Final:\n{current_state}\n")

agent = LocalSearchAgent(move_type.SWAP, first_improvement=True)
while True:
    action = agent.action_policy(current_state, env)
    if action is None:
        break
current_state = env.state_transition(current_state, action)
print(f"\nEstado Final:\n{current_state}")