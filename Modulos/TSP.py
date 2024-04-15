import random
from copy import deepcopy
import numpy as np

class TSP_Instance:
    def __init__(self, city_locations):
        self.city_locations = city_locations
        self.num_cities = len(city_locations)
        self.distance_matrix = np.sqrt(np.sum((city_locations[:, np.newaxis, :] -  city_locations[np.newaxis, :, :]) ** 2, axis=-1))

class TSP_State:
    def __init__(self, inst_info, visited=None):
        self.visited = visited if visited is not None else []
        self.not_visited = set(range(len(inst_info.distance_matrix))) - set(self.visited)
        self.is_complete = len(self.not_visited) == 0
        self.inst_info = inst_info
        self.cost = self.update_cost()  # Aquí se realiza el cálculo inicial del coste.

    def calculate_cost(self, visited):
        cost = 0
        if len(visited) > 1:
            for i in range(len(visited) - 1):
                cost += self.inst_info.distance_matrix[visited[i]][visited[i + 1]]
        return cost

    def update_cost(self):
        self.cost = self.calculate_cost(self.visited)
        return self.cost


    def __str__(self):
        return f"Tour actual: {self.visited}, \nCoste total: {self.cost}"

class TSP_Environment():
    @staticmethod
    def gen_actions(state, type, shuffle = False):
        if type == "constructive":
            actions = [("constructive", city) for city in state.not_visited]
        elif type == "2-opt":
            n = len(state.visited)
            actions = [(type, (i, j)) for i in range(n - 1) for j in range(i + 2, n-1)]

        else:
            raise NotImplementedError(f"Tipo de acción '{type}' no implementado")

        if shuffle:
            random.shuffle(actions)

        for action in actions:
            yield action


    @staticmethod
    def state_transition(state, action):
        # constructive-move: agrega una ciudad al tour
        if action[0]=="constructive" and state.is_complete==False:
            state.visited.append(action[1])
            state.not_visited.remove(action[1])

        if len(state.not_visited) == 0: # se completó el tour
            state.visited.append(state.visited[0])
            state.update_cost() #solo se actualiza en soluciones completas
            state.is_complete = True

        # 2-opt: intercambia dos aristas del tour
        elif action[0]=="2-opt" and state.is_complete==True:
            state.cost = TSP_Environment.calculate_cost_after_action(state, action)
            i, j = action[1]
            state.visited[i+1:j+1] = reversed(state.visited[i+1:j+1])

        else:
            raise NotImplementedError(f"Movimiento '{action}' no válido para estado {state}")

        return state

    def calculate_cost_after_action(state, action):
        if action[0] == "2-opt": #optimización para 2-opt
            visited = state.visited
            dist_matrix = state.inst_info.distance_matrix

            n = len(visited)
            i, j = action[1]
            dist_actual_i = dist_matrix[visited[i]][visited[(i+1)%n]]
            dist_actual_j = dist_matrix[visited[j]][visited[(j+1)%n]]
            nueva_dist_i = dist_matrix[visited[i]][visited[j]]
            nueva_dist_j = dist_matrix[visited[(i+1)%n]][visited[(j+1)%n]]

            # Calcular el cambio en el costo
            cambio_costo = (nueva_dist_i + nueva_dist_j) - (dist_actual_i + dist_actual_j)
            new_cost = state.cost + cambio_costo

        return new_cost

def evalConstructiveActions(state, env):
    evals = []
    for action in env.gen_actions(state, "constructive"):
        ultima_ciudad = state.visited[-1] if state.visited else 0
        eval = state.inst_info.distance_matrix[ultima_ciudad][action[1]]
        evals.append((action,1.0-eval))
    return evals

class GreedyAgent():
    def __init__(self, eval_actions):
        self.eval_actions= eval_actions

    def select_action(self, evals):
        return max(evals, key=lambda x: x[1])[0]

    def action_policy(self, state, env):
        evals = self.eval_actions(state, env)
        if len(evals)==0: return None

        # Seleccionar la acción que maximiza su evaluación
        return self.select_action(evals)

## Ejemplo

# creamos un problema con 20 ciudades en un plano 2D
cities  = np.random.rand(20, 2)
inst_info = TSP_Instance(cities)

# creamos un estado inicial
current_state = TSP_State (inst_info, visited=[0])
# referenciamos nuestro ambiente con las "reglas del juego"
env = TSP_Environment
# creamos nuestro agente
agent = GreedyAgent(evalConstructiveActions)

# iteramos hasta encontrar una solución
while not current_state.is_complete:
    action = agent.action_policy(current_state, env)
    current_state=env.state_transition(current_state, action)

print(current_state)