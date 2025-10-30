import enum
import numpy as np
import pygame
import os
from hopfield import HopfieldNetwork
from draw_utils import plot_avg_energies, save_gif_5x5, save_matrix_image, plot_accuracy, get_pattern_letter

def run_test():

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
        output = hf.recall(case)
        print("Input    :", case)
        print("Output   :", output)
        print("——————")

# ========== EJERICIO 2 ==========
def get_initial_patterns():
    """Retorna los patrones de las letras N, S, C y J"""
    N = np.array([
        1, -1, -1, -1,  1,
        1,  1, -1, -1,  1,
        1, -1,  1, -1,  1,
        1, -1, -1,  1,  1,
        1, -1, -1, -1,  1
    ])

    # Letra S
    S = np.array([
        1,  1,  1,  1,  1,
        1, -1, -1, -1, -1,
        1,  1,  1,  1,  1,
        -1, -1, -1, -1,  1,
        1,  1,  1,  1,  1
    ])

    # Letra C
    C = np.array([
        1,  1,  1,  1,  1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1,  1,  1,  1,  1
    ])

    # Letra J
    J = np.array([
        1,  1,  1,  1,  1,
        -1, -1, 1, -1,  -1,
        -1, -1, 1, -1,  -1,
        1, -1, 1, -1,  -1,
        1,  1,  1,  -1,  -1
    ])
    return [N, S, C, J]

def _final_state(recall_out):
    """
    Accept either a final state np.array or a trajectory (list/array),
    and always return the final np.array state (±1 entries).
    """
    if isinstance(recall_out, (list, tuple)):
        return np.array(recall_out[-1])
    recall_out = np.array(recall_out)
    # If recall returns shape (T, N) as a single ndarray trajectory:
    if recall_out.ndim == 2:
        return recall_out[-1]
    return recall_out

def _flip_bits(vec, k, rng):
    """Return a copy of vec with k randomly chosen bits flipped."""
    v = vec.copy()
    if k > 0:
        idx = rng.choice(v.size, size=k, replace=False)
        v[idx] *= -1
    return v

def accuracy_by_step(initial_patterns, rows, cols, trials_per_level=10, treat_negative_as_match=True, seed=42):
    """
    Compute accuracy vs. noise level (number of flipped bits).
    
    Args:
        initial_patterns: list/array of clean ±1 patterns (shape: [P, N]).
        rows, cols: grid shape so N = rows*cols.
        trials_per_level: how many random corruptions per (pattern, noise_bits).
        treat_negative_as_match: count -pattern as correct (common in Hopfield).
        seed: RNG seed for reproducibility.

    Returns:
        noise_bits: np.array of 0..N
        accuracy: np.array of length N+1 with mean accuracy at each noise level
        accuracy_per_pattern: np.array shape [P, N+1] with per-pattern accuracy
    """
    N = rows * cols
    P = len(initial_patterns)
    rng = np.random.default_rng(seed)

    # Will hold per-pattern accuracies across noise levels
    acc_per_pat = np.zeros((P, N + 1), dtype=float)

    for noise_bits in range(N + 1):
        for p_idx, base in enumerate(initial_patterns):
            correct = 0
            total = 0
            for _ in range(trials_per_level):
                noisy = _flip_bits(base, noise_bits, rng)
                # fresh Hopfield per trial, as in your original code
                hop = HopfieldNetwork(NETWORK_SIZE=N, starting_patterns=initial_patterns)
                recalled = _final_state(hop.recall(noisy))

                # Count a hit if equals base (or -base if enabled)
                match = np.array_equal(recalled, base)
                if treat_negative_as_match:
                    match = match or np.array_equal(recalled, -base)

                correct += 1 if match else 0
                total   += 1
            acc_per_pat[p_idx, noise_bits] = correct / max(total, 1)

    # Mean over patterns for the global curve
    acc_global = acc_per_pat.mean(axis=0)
    return np.arange(N + 1), acc_global, acc_per_pat

def avg_energy_by_step(get_initial_patterns, HopfieldNetwork, network_size=25, trials=100, max_iter=100, add_noise_bits=0, seed=123):
    """
    Calcula energía promedio por paso para cada patrón.
    - add_noise_bits: si >0, invierte aleatoriamente esa cantidad de bits al patrón inicial en cada corrida.
    Devuelve:
      energies_avg: lista de arrays (uno por patrón) con la energía promedio en cada paso.
      energies_trials: lista (por patrón) de listas (por corrida) con arrays de energía por paso.
    """
    rng = np.random.default_rng(seed)
    patterns = get_initial_patterns()
    hop = HopfieldNetwork(NETWORK_SIZE=network_size, starting_patterns=patterns)

    energies_avg = []
    energies_trials = []

    for p_idx, base in enumerate(patterns):
        trial_series = []

        for t in range(trials):
            x0 = base.copy()
            if add_noise_bits > 0:
                idx_flip = rng.choice(x0.size, size=add_noise_bits, replace=False)
                x0[idx_flip] *= -1

            _, E_series = hop.recall_and_energy(x0, max_iter=max_iter)
            trial_series.append(np.array(E_series, dtype=float))

        # Alinear longitudes rellenando con el último valor de cada serie
        max_len = max(len(s) for s in trial_series)
        aligned = []
        for s in trial_series:
            if len(s) < max_len:
                pad = np.full(max_len - len(s), s[-1])
                aligned.append(np.concatenate([s, pad]))
            else:
                aligned.append(s)
        aligned = np.stack(aligned, axis=0)        # shape: (trials, max_len)
        mean_per_step = aligned.mean(axis=0)       # energía promedio por paso

        energies_avg.append(mean_per_step)
        energies_trials.append(trial_series)

    return energies_avg, energies_trials

ROWS = 5
COLS = 5

def main():

    patterns = get_initial_patterns()  

    # noise_bits, acc, acc_per_pat = accuracy_by_step(
    #     initial_patterns=patterns,
    #     rows=ROWS,
    #     cols=COLS,
    #     trials_per_level=100,           # tiradas
    #     treat_negative_as_match=True,  # typical for Hopfield
    #     seed=123
    # )

    # plot_accuracy(noise_bits, acc, acc_per_pat,
    #               title=f"Tasa de aciertos vs. ruido")

    energies_avg, energies_trials = avg_energy_by_step(
    get_initial_patterns=get_initial_patterns,
    HopfieldNetwork=HopfieldNetwork,
    network_size=25,
    trials=100,
    max_iter=100,
    add_noise_bits=0,
    seed=42)
    plot_avg_energies(energies_avg)

if __name__ == "__main__":
    main()
