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
        for pattern in self.starting_patterns:
            for i in range(self.network_size):
                for j in range(self.network_size):
                    if i != j:
                        self.weights[i][j] += pattern[i] * pattern[j]
        # Normalizar los pesos dividiendo por el numero de patrones
        num_patterns = len(self.starting_patterns)
        for i in range(self.network_size):
            for j in range(self.network_size):
                self.weights[i][j] /= num_patterns

    def recall(self, pattern: np.array, max_iter=100):
        """
        Args:
            pattern (np.array): The input pattern to recall from the network. p = [1, -1, 1,...] of size NETWORK_SIZE
        """
        if len(pattern) != self.network_size:
            raise ValueError("Input pattern size must match the network size.")
        current_pattern = pattern.copy()
        for it in range(max_iter):
            changed = False
            for i in range(self.network_size):
                h_i = sum(self.weights[i][j] * current_pattern[j] for j in range(self.network_size))
                pattern_new_i = 1 if h_i >= 0 else -1
                if pattern_new_i != current_pattern[i]:
                    current_pattern[i] = pattern_new_i
                    changed = True
            if not changed:
                break
        return current_pattern