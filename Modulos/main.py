import numpy as np
from qap_instance import QAP_Instance
from qap_state import QAP_State
from qap_move_type import MoveType
from qap_environment import QAP_Environment
from qap_agents import GreedyAgent, LocalSearchAgent, eval_constructive_actions, ILS_Solver, Perturbation, SingleAgentSolver, TabuSearchSolver
from copy import deepcopy

flow = np.array([
    [0, 18, 23, 20, 34, 45, 47, 35, 25, 19, 30, 24, 26, 21, 29, 31, 40, 28, 38, 33],
    [18, 0, 17, 29, 35, 29, 38, 42, 36, 41, 32, 27, 25, 24, 22, 33, 40, 37, 44, 47],
    [23, 17, 0, 28, 32, 39, 40, 41, 37, 38, 36, 33, 29, 30, 32, 38, 42, 40, 46, 49],
    [20, 29, 28, 0, 17, 27, 33, 34, 25, 26, 19, 23, 30, 29, 27, 35, 28, 31, 38, 42],
    [34, 35, 32, 17, 0, 15, 22, 24, 18, 21, 12, 14, 26, 25, 22, 30, 22, 25, 30, 34],
    [45, 29, 39, 27, 15, 0, 12, 14, 23, 27, 19, 21, 33, 32, 30, 36, 20, 21, 28, 32],
    [47, 38, 40, 33, 22, 12, 0, 10, 18, 24, 17, 20, 29, 28, 25, 32, 17, 19, 24, 28],
    [35, 42, 41, 34, 24, 14, 10, 0, 19, 25, 18, 20, 28, 27, 24, 31, 15, 16, 22, 26],
    [25, 36, 37, 25, 18, 23, 18, 19, 0, 15, 12, 10, 21, 20, 18, 25, 13, 10, 18, 22],
    [19, 41, 38, 26, 21, 27, 24, 25, 15, 0, 9, 12, 18, 17, 14, 22, 11, 5, 12, 16],
    [30, 32, 36, 19, 12, 19, 17, 18, 12, 9, 0, 6, 16, 15, 13, 20, 11, 8, 15, 19],
    [24, 27, 33, 23, 14, 21, 20, 20, 10, 12, 6, 0, 13, 12, 10, 17, 8, 5, 11, 15],
    [26, 25, 29, 30, 26, 33, 29, 28, 21, 18, 16, 13, 0, 3, 7, 11, 20, 18, 25, 29],
    [21, 24, 30, 29, 25, 32, 28, 27, 20, 17, 15, 12, 3, 0, 4, 8, 17, 15, 22, 26],
    [29, 22, 32, 27, 22, 30, 25, 24, 18, 14, 13, 10, 7, 4, 0, 7, 14, 12, 18, 22],
    [31, 33, 38, 35, 30, 36, 32, 31, 25, 22, 20, 17, 11, 8, 7, 0, 13, 11, 18, 22],
    [40, 40, 42, 28, 22, 20, 17, 15, 13, 11, 11, 8, 20, 17, 14, 13, 0, 3, 10, 14],
    [28, 37, 40, 31, 25, 21, 19, 16, 10, 5, 8, 5, 18, 15, 12, 11, 3, 0, 7, 11],
    [38, 44, 46, 38, 30, 28, 24, 22, 18, 12, 15, 11, 25, 22, 18, 18, 10, 7, 0, 4],
    [33, 47, 49, 42, 34, 32, 28, 26, 22, 16, 19, 15, 29, 26, 22, 22, 14, 11, 4, 0]
])

distance = np.array([
    [0, 19, 25, 21, 28, 30, 32, 35, 30, 25, 20, 18, 22, 24, 27, 29, 34, 38, 40, 42],
    [19, 0, 18, 15, 20, 22, 24, 26, 28, 29, 23, 21, 20, 22, 25, 28, 33, 37, 39, 41],
    [25, 18, 0, 20, 24, 26, 28, 30, 28, 24, 19, 17, 15, 17, 20, 23, 28, 32, 34, 36],
    [21, 15, 20, 0, 17, 19, 21, 23, 25, 26, 20, 18, 16, 18, 21, 24, 29, 33, 35, 37],
    [28, 20, 24, 17, 0, 15, 17, 19, 21, 23, 18, 16, 14, 16, 19, 22, 27, 31, 33, 35],
    [30, 22, 26, 19, 15, 0, 16, 18, 20, 21, 16, 14, 12, 14, 17, 20, 25, 29, 31, 33],
    [32, 24, 28, 21, 17, 16, 0, 15, 17, 18, 13, 11, 9, 11, 14, 16, 21, 25, 27, 29],
    [35, 26, 30, 23, 19, 18, 15, 0, 19, 20, 15, 13, 11, 13, 16, 18, 23, 27, 29, 31],
    [30, 28, 28, 25, 21, 20, 17, 19, 0, 14, 10, 8, 10, 12, 15, 18, 23, 27, 29, 31],
    [25, 29, 24, 26, 23, 21, 18, 20, 14, 0, 12, 10, 12, 14, 17, 20, 25, 29, 31, 33],
    [20, 23, 19, 20, 18, 16, 13, 15, 10, 12, 0, 9, 11, 13, 16, 19, 24, 28, 30, 32],
    [18, 21, 17, 18, 16, 14, 11, 13, 8, 10, 9, 0, 8, 10, 13, 16, 21, 25, 27, 29],
    [22, 20, 15, 16, 14, 12, 9, 11, 10, 12, 11, 8, 0, 9, 12, 15, 20, 24, 26, 28],
    [24, 22, 17, 18, 16, 14, 11, 13, 12, 14, 13, 10, 9, 0, 11, 14, 19, 23, 25, 27],
    [27, 25, 20, 21, 19, 17, 14, 16, 15, 17, 16, 13, 12, 11, 0, 13, 18, 22, 24, 26],
    [29, 28, 23, 24, 22, 20, 16, 18, 18, 20, 19, 16, 15, 14, 13, 0, 16, 20, 22, 24],
    [34, 33, 28, 29, 27, 25, 21, 23, 23, 25, 24, 21, 20, 19, 18, 16, 0, 14, 16, 18],
    [38, 37, 32, 33, 31, 29, 25, 27, 27, 29, 28, 25, 24, 23, 22, 20, 14, 0, 12, 14],
    [40, 39, 34, 35, 33, 31, 27, 29, 29, 31, 30, 27, 26, 25, 24, 22, 16, 12, 0, 2],
    [42, 41, 36, 37, 35, 33, 29, 31, 31, 33, 32, 29, 28, 27, 26, 24, 18, 14, 2, 0]
])

env = QAP_Environment()
instance = QAP_Instance(flow, distance)
current_state = QAP_State(instance)

greedy = GreedyAgent(eval_constructive_actions)
while True:
    action = greedy.action_policy(current_state, env)
    if action is None:
        break
    current_state = env.state_transition(current_state, action)
print(f"\nResultado Greedy:\n{current_state}\n")

local_search = LocalSearchAgent(MoveType.SWAP, first_improvement=True)
current_cost = current_state.cost
while True:
    action = local_search.action_policy(current_state, env)
    if action is None:
        break
    current_state = env.state_transition(current_state, action)
    
    new_cost = current_state.cost
    if new_cost >= current_cost:
        break
    current_cost = new_cost
print(f"\nResultado LocalSearchAgent:\n{current_state}\n")

greedy = SingleAgentSolver(env, GreedyAgent(eval_constructive_actions))
local_search = SingleAgentSolver(env, LocalSearchAgent(MoveType.SWAP, first_improvement=True))

current_state = QAP_State(instance)
current_state, *_ = greedy.solve(current_state)
current_state, *_ = local_search.solve(current_state)
print(f"\nResultado SingleAgentSolver:\n{current_state}\n")

initial_state = QAP_State(instance, assigned=[0])
ils_solver = ILS_Solver(greedy, local_search, Perturbation(MoveType.SWAP, pert_size=2))
current_state, _, iterations = ils_solver.solve(initial_state, save_history=True, max_actions=100)
print(f"\nResultado ILS:\n{current_state}\n")

tabusearch = SingleAgentSolver(env, TabuSearchSolver(MoveType.SWAP, tabu_size=5, budget=100))
ts_state, history_ts5 = tabusearch.solve(deepcopy(initial_state),
                                save_history=True, track_best_state=True)
print("Resultado TabuSearch:\n", ts_state)

#class QAP_Instance_Generator:
#    @staticmethod
#    def generate_matrix(n, lower_limit, upper_limit):
#        matrix = np.random.randint(lower_limit, upper_limit + 1, size=(n, n))
#        matrix = np.triu(matrix)
#
#        matrix += matrix.T - np.diag(np.diag(matrix))
#
#        np.fill_diagonal(matrix, 0)
#
#        return matrix
#
#    @staticmethod
#    def generate_instance(n, flow_limit=30, distance_limit=1000, seed=None):
#        if seed is not None:
#            np.random.seed(seed)
#
#        if n < 3:
#            n = 3
#
#        flow_matrix = QAP_Instance_Generator.generate_matrix(n, 1, flow_limit)
#        distance_matrix = QAP_Instance_Generator.generate_matrix(n, 1, distance_limit)
#
#        instance = QAP_Instance(flow_matrix, distance_matrix)
#        return instance
#
#def run_qap_algorithms(greedy, local_search, ils_solver, tabusearch, n=10, flow_limit=10, distance_limit=100):
#    greedy_results = []
#    sls_results = []
#    ils_results = []
#    ts_results = []
#    
#    for i in range(10):
#        print(f"Iteration {i} with size {n}")
#        qap_instance = QAP_Instance_Generator.generate_instance(n, flow_limit, distance_limit)
#        current_state = QAP_State(qap_instance, assigned=[0])
#
#        current_state, *_ = greedy.solve(current_state)
#        greedy_results.append(current_state.cost)
#
#        current_state, *_ = local_search.solve(current_state)
#        sls_results.append(current_state.cost)
#
#        current_state = QAP_State(qap_instance, assigned=[0])
#        current_state, *_ = ils_solver.solve(current_state)
#        ils_results.append(current_state.cost)
#
#        current_state = QAP_State(qap_instance, assigned=[0])
#        current_state, *_ = tabusearch.solve(current_state)
#        ts_results.append(current_state.cost)
#
#        return greedy_results, sls_results, ils_results, ts_results
#
#def calculate_average_results(greedy, local_search, ils_solver, tabusearch, sizes):
#    average_results_greedy = []
#    average_results_sls = []
#    average_results_ils = []
#    average_results_ts = []
#    for n in sizes:
#        greedy_results, sls_results, ils_results, ts_results = \
#        run_qap_algorithms(greedy, local_search, ils_solver, tabusearch, n, flow_limit=10, distance_limit=100)
#        average_greedy = sum(greedy_results) / len(greedy_results)
#        average_sls = sum(sls_results) / len(sls_results)
#        average_ils = sum(ils_results) / len(ils_results)
#        average_ts = sum(ts_results) / len(ts_results)
#        average_results_greedy.append(average_greedy)
#        average_results_sls.append(average_sls)
#        average_results_ils.append(average_ils)
#        average_results_ts.append(average_ts)
#    return average_results_greedy, average_results_sls, average_results_ils, average_results_ts
#
#sizes = [10, 20, 30, 40]
#
#greedy = SingleAgentSolver(env, GreedyAgent(eval_constructive_actions))
#local_search = SingleAgentSolver(env, LocalSearchAgent(MoveType.SWAP, first_improvement=True))
#ils = ILS_Solver(greedy, local_search, Perturbation(MoveType.SWAP, pert_size=2))
#
#ts_local_search = SingleAgentSolver(env, LocalSearchAgent(None, first_improvement=True))
#ts = TabuSearchSolver(env, greedy, MoveType.SWAP)
#
#average_results_greedy, average_results_sls, average_results_ils, average_results_ts = \
#    calculate_average_results(greedy, local_search, ils, ts, sizes)