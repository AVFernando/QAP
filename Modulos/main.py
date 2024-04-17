import numpy as np
from qap_instance import QAP_Instance
from qap_state import QAP_State
from qap_move_type import MoveType
from qap_environment import QAP_Environment
from qap_agents import GreedyAgent, LocalSearchAgent, eval_constructive_actions, ILS_Solver, Perturbation, SingleAgentSolver

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
print(f"\nResultado LSA:\n{current_state}\n")

greedy = SingleAgentSolver(env, GreedyAgent(eval_constructive_actions))
local_search = SingleAgentSolver(env, LocalSearchAgent(MoveType.SWAP, first_improvement=True))

current_state = QAP_State(instance)
current_state, *_ = greedy.solve(current_state)
current_state, *_ = local_search.solve(current_state)
print(f"\nResultado SAS:\n{current_state}\n")

initial_state = QAP_State(instance, assigned=[0])

ils_solver = ILS_Solver(greedy, local_search, Perturbation(MoveType.SWAP, pert_size=2))
current_state, _, iterations = ils_solver.solve(initial_state, save_history=True, max_actions=100)
print(f"\nResultado ILS:\n{current_state}\n")