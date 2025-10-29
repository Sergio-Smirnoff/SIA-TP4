from enum import Enum
import numpy as np

class State(Enum):
    ACTIVE = 1
    INACTIVE = -1

class HopfieldNeuron:
    def __init__(self, NETWORK_SIZE: int, state=State.INACTIVE):
        self.network_size = NETWORK_SIZE
        self.state = state

    def update_state(self, new_state: State):
        self.state = new_state

    def activation_function(self, activation_sum)-> int:
        h_i = activation_sum
        if h_i > 0:
            return State.ACTIVE.value
        elif h_i < 0:
            return State.INACTIVE.value
        else:
            raise ValueError("Activation resulted in zero, neuron musnt update!!!!.")



class HopfieldNetwork:
    def __init__(self, NETWORK_SIZE: int, starting_patterns):
        """
        Args:
            NETWORK_SIZE (int): The size of the Hopfield network.
            starting_patterns (list): A list of initial patterns to store in the network. [pattern1, pattern2, ...]
                where each pattern: pattern = [1, -1, 1, ...] of size NETWORK_SIZE
        """
        self.network_size = NETWORK_SIZE
        self.neurons = [HopfieldNeuron(NETWORK_SIZE, State.INACTIVE) for _ in range(NETWORK_SIZE)]
        self.starting_patterns = starting_patterns
        #starting_patterns = [ 
        #                       [1, -1, 1,..], 
        #                       [-1, 1, -1,..], 
        #                     ...]

        #matriz de pesos de tamanio NETWORK_SIZE x NETWORK_SIZE
        #cada elemento w_ij representa la fuerza de la conexion entre la neurona i y j
        self.weights = [[0.0 for _ in range(NETWORK_SIZE)] for _ in range(NETWORK_SIZE)]
        self.initialize_weights()
        #TODO como inicializar con patrones predefinidos


    def initialize_weights(self):
        N = self.network_size
        W = np.zeros((N, N), dtype=float)
        # patterns come as lists/np arrays of shape (N,)
        for pattern in self.starting_patterns:
            v = np.asarray(pattern, dtype=float).reshape(N, 1)
            W += v @ v.T
        np.fill_diagonal(W, 0.0)  # no self-connections
        W /= N
        self.weights = W.tolist()

    def recall(self, pattern: np.array, max_iter=100):
        """
        Args:
            pattern (np.array): The input pattern to recall from the network. p = [1, -1, 1,...] of size NETWORK_SIZE

        Returns:
            np.array: list of results for each step.
            
        """
        if len(pattern) != self.network_size:
            raise ValueError("Input pattern size must match the network size.")
        results_by_step = []
        results_by_step.append(pattern.copy())  #dejo el primero para graficarlo
        current_pattern = pattern.copy()
        for it in range(max_iter):
            changed = False
            for i in range(self.network_size):
                h_i = sum(self.weights[i][j] * current_pattern[j] for j in range(self.network_size))
                pattern_new_i = 1 if h_i >= 0 else -1
                if pattern_new_i != current_pattern[i]:
                    current_pattern[i] = pattern_new_i
                    changed = True
            results_by_step.append(current_pattern.copy())
            if not changed:
                break
        
        #return all results by step
        #access last result with results_by_step[-1]
        return results_by_step
    

    def recall_and_energy(self, pattern: np.array, max_iter=100):
        """
        Realiza recall asíncrono y registra energía en cada paso (incluye estado inicial).
        Devuelve:
        results_by_step: [s^0, s^1, ..., s^T]
        energy_by_step:  [E(s^0), E(s^1), ..., E(s^T)]
        """
        if len(pattern) != self.network_size:
            raise ValueError("Input pattern size must match the network size.")
        
        results_by_step = []
        energy_by_step = []

        current = pattern.copy()
        results_by_step.append(current.copy())

        # energía inicial E = -0.5 * s^T W s
        E = -0.5 * np.sum(self.weights * np.outer(current, current))
        energy_by_step.append(E)

        for _ in range(max_iter):
            changed = False
            for i in range(self.network_size):
                h_i = np.dot(self.weights[i], current)
                new_state = 1 if h_i >= 0 else -1
                if new_state != current[i]:
                    current[i] = new_state
                    changed = True

            results_by_step.append(current.copy())
            E = -0.5 * np.sum(self.weights * np.outer(current, current))
            energy_by_step.append(E)

            if not changed:
                break

        return results_by_step, energy_by_step