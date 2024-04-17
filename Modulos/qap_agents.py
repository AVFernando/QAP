import numpy as np
import random
from copy import deepcopy
from enum import Enum
from qap_move_type import MoveType

def eval_constructive_actions(state, env):
    evals = []
    n = len(state.assigned)
    for action in env.gen_actions(state, MoveType.CONSTRUCTIVE):
        place2 = action[1]
        if place2 in state.assigned:
            continue
        cost = sum(state.instance.flow_matrix[place1, place2] * state.instance.distance_matrix[i, n] for i, place1 in enumerate(state.assigned))
        evals.append((action, -cost))
    return evals

class GreedyAgent:
    def __init__(self, eval_actions):
        self.eval_actions = eval_actions
    
    def reset(self):
        pass

    def select_action(self, evals):
        return max(evals, key=lambda x: x[1])[0]

    def action_policy(self, state, env):
        evals = self.eval_actions(state, env)
        return self.select_action(evals) if evals else None

    def __deepcopy__(self, memo):
        return type(self)(eval_actions=self.eval_actions)

class LocalSearchAgent(GreedyAgent):
    def __init__(self, action_type, first_improvement=True):
        self.env = None
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

class SingleAgentSolver():
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def solve(self, state, track_best_state=False, save_history=False, max_actions=0):
        history = None

        if save_history:
            history = [(None, state.cost)]

        if max_actions == 0:
            max_actions = 99999999

        best_state = None

        if track_best_state:
            best_state = deepcopy(state)

        self.agent.reset()

        n_actions = 0
        while n_actions < max_actions:
            action = self.agent.action_policy(state, self.env)
            if action is None:
                break

            state = self.env.state_transition(state, action)
            n_actions += 1

            if track_best_state and state.cost < best_state.cost:
                best_state = deepcopy(state)

            if save_history:
                history.append((action, state.cost))

        if track_best_state:
            return best_state, history
        else:
            return state, history, n_actions

    def multistate_solve(self, states, track_best_state=False, save_history=False, max_actions=0):
        agents = [deepcopy(self.agent) for _ in range(len(states))]
        history = [None]*len(states)
        best_state = [None]*len(states)
        n_actions = [None]*len(states)

        if max_actions == 0:
            max_actions = 99999999

        for i in range(len(states)):
            agents[i].reset()
            n_actions[i] = 0
            history[i] = []
            if track_best_state: best_state[i] = deepcopy(states[i])

        live_states_idx = list(range(len(states)))

        for _ in range(max_actions):
            evals = agents[0].eval_actions([states[i] for i in live_states_idx], self.env)

            new_idx = []
            for i in live_states_idx:
                eval = evals[live_states_idx.index(i)]

                if eval == []:
                    continue

                action = agents[i].select_action(eval)

                states[i] = self.env.state_transition(states[i], action)
                n_actions[i] += 1

                new_idx.append(i)

                if track_best_state and states[i].cost < best_state.cost:
                    best_state[i] = deepcopy(states[i])

                if save_history:
                    history[i].append((action, states[i].cost))

            live_states_idx = new_idx

            if new_idx == []:
                break

        if track_best_state:
            return best_state, history
        else:
            return states, history, n_actions

class Perturbation:
    def __init__(self, action_type, pert_size=3):
        self.pert_size = pert_size
        self.action_type = action_type

    def __call__(self, state, env):
        gen_action = env.gen_actions(state, self.action_type, shuffle=True)
        for _ in range(self.pert_size):
            action = next(gen_action)
            env.state_transition(state, action)
        return state

def default_acceptance_criterion(min_cost, new_cost):
    return new_cost <= min_cost

class ILS_Solver:
    def __init__(self, constructive_solver, local_search_solver, perturbation,
        acceptance_criterion=default_acceptance_criterion):
        self.constructive_solver = constructive_solver
        self.local_search_solver = local_search_solver
        self.perturbation = perturbation
        self.acceptance_criterion = acceptance_criterion
        self.env = local_search_solver.env

    def solve(self, initial_state, save_history=False, max_actions=0):
        history = None

        current_solution, *_ = self.constructive_solver.solve(initial_state)
        current_solution, _history, n_act = self.local_search_solver.solve(current_solution, save_history=save_history)

        if save_history:
            history = _history
        n_actions = n_act

        best_solution = deepcopy(current_solution)

        while n_actions < max_actions:
            perturbed_solution = self.perturbation(deepcopy(current_solution), self.env)
            perturbed_solution.update_cost()

            local_optimum, hist, n_act = self.local_search_solver.solve(perturbed_solution, save_history=save_history, max_actions=max_actions-n_actions)

            if save_history:
                history += hist
            n_actions += n_act

            cost = local_optimum.cost

            if self.acceptance_criterion(best_solution.cost, cost):
                current_solution = local_optimum

                if cost < best_solution.cost:
                    best_solution = deepcopy(current_solution)

        return best_solution, history, n_actions