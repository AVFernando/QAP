import random
import numpy as np

class QAP_Instance:
    def __init__(self):
        n = 3 # Tamaño de las matrices
        self.n = n
        self.flow_matrix = np.random.randint(1, 10, size=(n, n)) # Matriz de flujo
        self.distance_matrix = np.random.randint(1, 10, size=(n, n)) # Matriz de distancia
        self.assignment_matrix = np.zeros((n, n)) # Matriz de asignacion (inicializada a 0)
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
        self.not_assigned = set(range(len(inst_info.assignment_matrix))) - set(self.assigned)
        self.all_assigned = len(self.not_assigned) == 0
        self.inst_info = inst_info
        self.cost = self.calculate_cost()

    def calculate_cost(self):
        cost_matrix = np.multiply(self.inst_info.distance_matrix, self.inst_info.assignment_matrix)
        cost = np.sum(np.multiply(self.inst_info.flow_matrix, cost_matrix))
        return int(cost)

class QAP_Environment():
    @staticmethod
    def gen_actions(state, type, shuffle=False):
        if type == "constructive":
            actions = [("constructive", assigned) for assigned in state.not_assigned]
        elif type == "swap":
            n_len = len(state.assigned)
            actions = [(type, (i, j)) for i in range(n_len) for j in range(i + 1, n_len)]
        else:
            raise NotImplementedError(f"Tipo de accion '{type}' no implementado para QAP")

        if shuffle:
            random.shuffle(actions)

        for action in actions:
            yield action

    @staticmethod
    def state_transition(state, action):
        new_state = QAP_State(state.inst_info, assigned=state.assigned.copy())

        if action[0] == "constructive" and not state.all_assigned:
            new_state.assigned.append(action[1])
            new_state.not_assigned.remove(action[1])
            new_state.inst_info.assignment_matrix[action[1], action[1]] = 1
            new_state.cost = new_state.calculate_cost()

        elif len(state.not_assigned) == 0:
            new_state.assigned.append(state.assigned[0])
            new_state.inst_info.assignment_matrix[state.assigned[0], state.assigned[0]] = 1
            new_state.cost = new_state.calculate_cost()
            new_state.all_assigned = True

        elif action[0] == "swap":
            i, j = action[1]
            if i not in new_state.assigned or j not in new_state.assigned:
                raise ValueError(f"Las instalaciones {i} y/o {j} no están asignadas.")

            i_idx = new_state.assigned.index(i)
            j_idx = new_state.assigned.index(j)
            new_state.assigned[i_idx] = j
            new_state.assigned[j_idx] = i
            new_state.inst_info.assignment_matrix[i, j] = 1
            new_state.inst_info.assignment_matrix[j, i] = 1
            new_state.cost = new_state.calculate_cost()

        else:  # Acción no válida
            raise NotImplementedError(f"Tipo de acción '{action[0]}' no implementado para QAP")

        return new_state



    def calculate_cost_after_action(state, assigned):
        temp_assignment_matrix = np.copy(state.inst_info.assignment_matrix)
        not_assigned_locations = np.where(temp_assignment_matrix[:, assigned] == 0)[0]
        if len(not_assigned_locations) == 0:
            raise ValueError(f"No hay ubicaciones disponibles para la instalacion {assigned}.")
        
        temp_assignment_matrix[assigned, not_assigned_locations[0]] = 1
        temp_state = QAP_State(state.inst_info, assigned=state.assigned + [assigned])
        temp_state.inst_info.assignment_matrix = temp_assignment_matrix
        cost = temp_state.calculate_cost()
        
        return cost

def evalConstructiveActions(state, env):
    evals = []
    for action in env.gen_actions(state, "constructive"):
        cost_after_action = env.calculate_cost_after_action(state, action[1])
        evals.append((action, cost_after_action))
    return evals

class GreedyAgent():
    def __init__(self, evals):
        self.evals = evals

    def select_action(self, evals):
        return max(evals, key=lambda x: x[1])[0]

    def action_policy(self, state, env):
        evals = self.evals(state, env)
        if len(evals)==0: return None

        # Seleccionar la acción que maximiza su evaluación
        return self.select_action(evals)

# Crear una instancia del problema QAP
inst_info = QAP_Instance()
# Crear un estado inicial
current_state = QAP_State(inst_info, assigned=[])
# Crear el entorno
env = QAP_Environment
# Crear un agente Greedy
greedy_agent = GreedyAgent(evalConstructiveActions)

# Imprimir el estado inicial
print("\nEstado inicial:")
print("Instalaciones asignadas:", current_state.assigned)
print("Instalaciones no asignadas:", current_state.not_assigned)
print("Costo inicial:", current_state.cost)

# Actuar con el agente Greedy
while not current_state.all_assigned:
    action = greedy_agent.action_policy(current_state, env)
    current_state = env.state_transition(current_state, action)
    print("\nAccion tomada:", action)
    print("Instalaciones asignadas:", current_state.assigned)
    print("Instalaciones no asignadas:", current_state.not_assigned)
    print("Costo despues de la accion:", current_state.cost)

print("\nEstado final:")
print("Instalaciones asignadas:", current_state.assigned)
print("Instalaciones no asignadas:", current_state.not_assigned)
print("Costo final:", current_state.cost)