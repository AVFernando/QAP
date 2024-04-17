import numpy as np
import random
from copy import deepcopy
from enum import Enum
from qap_move_type import MoveType

class QAP_Environment:
    @staticmethod
    def gen_actions(state, move_type, shuffle=False):
        if move_type == MoveType.CONSTRUCTIVE:
            actions = [(MoveType.CONSTRUCTIVE, place) for place in state.not_assigned]
            
        elif move_type == MoveType.SWAP:
            
            actions = [(MoveType.SWAP, (i, j)) for i in range(len(state.assigned)) for j in range(i + 1, len(state.assigned))]
        elif move_type == MoveType.TWO_OPT:
            
            actions = [(MoveType.TWO_OPT, (i, j)) for i in range(len(state.assigned)) for j in range(i - 1, i + 2) if j != i]
            
        elif move_type == MoveType.RELOCATION:
            actions = [(MoveType.RELOCATION, (i, j)) for i in range(len(state.assigned)) for j in range(len(state.assigned)) if i != j]
        
        if shuffle:
            random.shuffle(actions)
        
        for action in actions:
            yield action

        return actions

    @staticmethod
    def state_transition(state, action):
        if action[0] == MoveType.CONSTRUCTIVE and not state.all_assigned:
            state.assigned.append(action[1])
            state.not_assigned.remove(action[1])
            state.all_assigned = len(state.not_assigned) == 0
            state.update_cost()
        elif action[0] == MoveType.SWAP and state.all_assigned:
            i, j = action[1]
            state.assigned[i], state.assigned[j] = state.assigned[j], state.assigned[i]
            state.update_cost()
        elif action[0] == MoveType.TWO_OPT and state.all_assigned:
            i, j = action[1]
            i_idx, j_idx = state.assigned.index(i), state.assigned.index(j)
            state.assigned[i_idx], state.assigned[j_idx] = j, i
            state.update_cost()
        elif action[0] == MoveType.RELOCATION and state.all_assigned:
            i, j = action[1]
            elem = state.assigned.pop(i)
            state.assigned.insert(j, elem)
            state.update_cost()
        else:
            raise NotImplementedError(f"Tipo de accion '{action[0]}' no implementado para QAP")
        
        return state

    @staticmethod
    def calculate_cost_after_action(state, action):
        if action[0] == MoveType.SWAP:
            k, l = action[1]
            state.assigned[k], state.assigned[l] = state.assigned[l], state.assigned[k]
            new_cost = sum(state.calculate_cost(i, j) for i in range(len(state.assigned)) for j in range(len(state.assigned)) if i != j)
            state.assigned[l], state.assigned[k] = state.assigned[k], state.assigned[l]
            return new_cost
        
        return state.cost
