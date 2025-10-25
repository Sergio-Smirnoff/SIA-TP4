import numpy as np
from hopfield import HopfieldNetwork

def run_test(input_pattern, network):
    output = network.recall(input_pattern)
    print("Input    :", input_pattern)
    print("Output   :", output)
    print("——————")

def main():
    patterns = [
        np.array([ 1, -1,  1, -1]),
        np.array([-1,  1, -1,  1]),
    ]
    hf = HopfieldNetwork(NETWORK_SIZE=4, starting_patterns=patterns)

    test_cases = [
        np.array([ 1, -1,  1, -1]),  # caso 1
        np.array([-1,  1, -1,  1]),  # caso 2
        np.array([ 1, -1, -1, -1]),  # caso 3
        np.array([ 1,  1,  1, -1]),  # caso 4
        np.array([ 1,  1, -1, -1]),  # caso 5
        np.array([ 1,  1, -1,  1]),  # caso 6 variante
    ]

    for case in test_cases:
        run_test(case, hf)

if __name__ == "__main__":
    main()
