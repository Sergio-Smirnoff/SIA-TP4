from enum import Enum
import numpy as np

class State(Enum):
    ACTIVE = 1
    INACTIVE = -1

class HopfieldNeuron:
    def __init__(self, NETWORK_SIZE: int, state=State.INACTIVE):
        self.network_size = NETWORK_SIZE
        self.state = state 
        self.weights = [0.0 for i in range(NETWORK_SIZE-1)] #a weight relating this neuron to each other neuron in the network


    # S_i = sign(activation_sum)
    def update_state(self, activation_sum):
        if activation_sum == 0:
            return  # No update if activation sum is zero
        self.state = State.ACTIVE if activation_sum > 0 else State.INACTIVE


    def activation_function(self, activation_sum)-> int:
        h_i = activation_sum
        if h_i > 0:
            return State.ACTIVE.value
        elif h_i < 0:
            return State.INACTIVE.value
        else:
            raise ValueError("Activation resulted in zero, neuron musnt update!!!!.")

    def initialize_weights(self, weights: np.array):  #TODO check weights range
        self.weights = weights



class HopfieldNetwork:
    def __init__(self, NETWORK_SIZE: int):
        self.network_size = NETWORK_SIZE
        self.neurons = [HopfieldNeuron(NETWORK_SIZE, State.INACTIVE) for _ in range(NETWORK_SIZE)]


    def set_weights(self, weight_matrix: np.ndarray):
        """Set the weights for each neuron in the network.
        Args:
            weight_matrix (np.ndarray): weight matrix NxN dimension.

        """
        for i, neuron in enumerate(self.neurons):
            pass

    def train(self, patterns: np.array):
        pass