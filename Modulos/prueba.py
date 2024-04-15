import random
import numpy as np

class QAP_Instance:
    def __init__(self):
        n = 3
        self.n = n
        self.flow_matrix = np.random.randint(1, 10, size=(n, n))
        self.distance_matrix = np.random.randint(1, 10, size=(n, n))
        self.assignment_matrix = np.zeros((n, n))
        print("\nMatrices generadas aleatoriamente:")
        print("Matriz de flujo:")
        print(self.flow_matrix)
        print("\nMatriz de distancia:")
        print(self.distance_matrix)
        print("\nMatriz de asignacion:")
        print(self.assignment_matrix)

class QAP_State:
    def __init__(self, inst_info, assigned=None):
        self.assigned = assigned if assigned is not None else []
        self.not_assigned = np.where(inst_info.assignment_matrix == 0)[0]
        self.all_assigned = len(self.not_assigned) == 0
        self.inst_info = inst_info
        self.cost = self.calculate_cost()

    def calculate_cost(self):
        cost_matrix = np.multiply(self.inst_info.distance_matrix, self.inst_info.assignment_matrix)
        cost = np.sum(np.multiply(self.inst_info.flow_matrix, cost_matrix))
        return int(cost)

class QAP_Environment:
    @staticmethod
    def gen_actions(state, type, shuffle=False):
        if type == "constructive":
            actions = [("constructive", assigned) for assigned in state.not_assigned]
        elif type == "swap":
            n_assigned = len(state.assigned)
            actions = [(type, (i, j)) for i in range(n_assigned) for j in range(i + 1, n_assigned)]
        else:
            raise NotImplementedError(f"Tipo de acción '{type}' no implementado para QAP")

        if shuffle:
            random.shuffle(actions)

        for action in actions:
            yield action

    @staticmethod
    def state_transition(state, action):
        if action[0] == "constructive" and not state.all_assigned:
            state.assigned.append(action[1])
            state.inst_info.assignment_matrix[action[1], state.not_assigned[0]] = 1
            state.not_assigned = np.where(state.inst_info.assignment_matrix == 0)[0]
            state.all_assigned = len(state.not_assigned) == 0
        elif action[0] == "swap":
            i, j = action[1]
            if i not in state.assigned or j not in state.assigned:
                raise ValueError(f"Las instalaciones {i} y/o {j} no están asignadas.")
            
            # Intercambiar instalaciones en la lista de asignadas
            i_idx = state.assigned.index(i)
            j_idx = state.assigned.index(j)
            state.assigned[i_idx], state.assigned[j_idx] = state.assigned[j_idx], state.assigned[i_idx]

            # Intercambiar filas y columnas en la matriz de asignación
            state.inst_info.assignment_matrix[i, :], state.inst_info.assignment_matrix[j, :] = state.inst_info.assignment_matrix[j, :].copy(), state.inst_info.assignment_matrix[i, :].copy()
            state.inst_info.assignment_matrix[:, i], state.inst_info.assignment_matrix[:, j] = state.inst_info.assignment_matrix[:, j].copy(), state.inst_info.assignment_matrix[:, i].copy()

            state.cost = state.calculate_cost()
        else:
            raise NotImplementedError(f"Tipo de accion '{action[0]}' no implementado para QAP")

    @staticmethod
    def calculate_cost_after_constructive(state, assigned):
        temp_assignment_matrix = np.copy(state.inst_info.assignment_matrix)
        not_assigned_locations = np.where(temp_assignment_matrix[:, assigned] == 0)[0]
        if len(not_assigned_locations) == 0:
            raise ValueError(f"No hay ubicaciones disponibles para la instalación {assigned}.")
        
        temp_assignment_matrix[assigned, not_assigned_locations[0]] = 1
        temp_state = QAP_State(state.inst_info, assigned=state.assigned + [assigned])
        temp_state.inst_info.assignment_matrix = temp_assignment_matrix
        cost = temp_state.calculate_cost()
        
        return cost

class GreedyAgent:
    def __init__(self, env):
        self.env = env

    def act(self, state):
        best_action = None
        best_cost = float("inf")
        for action in self.env.gen_actions(state, "constructive"):
            cost_after_action = self.env.calculate_cost_after_constructive(state, action[1])
            if cost_after_action < best_cost:
                best_action = action
                best_cost = cost_after_action
        return best_action

# Crear una instancia del problema QAP
qap_instance = QAP_Instance()

# Crear un estado inicial
initial_state = QAP_State(qap_instance)

# Crear el entorno
env = QAP_Environment()

# Crear un agente Greedy
greedy_agent = GreedyAgent(env)

# Imprimir el estado inicial
print("\nEstado inicial:")
print("Instalaciones asignadas:", initial_state.assigned)
print("Instalaciones no asignadas:", initial_state.not_assigned)
print("Costo inicial:", initial_state.cost)

# Actuar con el agente Greedy
while not initial_state.all_assigned:
    action = greedy_agent.act(initial_state)
    env.state_transition(initial_state, action)
    print("\nAcción tomada:", action)
    print("Instalaciones asignadas:", initial_state.assigned)
    print("Instalaciones no asignadas:", initial_state.not_assigned)
    print("Costo después de la acción:", initial_state.cost)

print("\nEstado final:")
print("Instalaciones asignadas:", initial_state.assigned)
print("Instalaciones no asignadas:", initial_state.not_assigned)
print("Costo final:", initial_state.cost)
