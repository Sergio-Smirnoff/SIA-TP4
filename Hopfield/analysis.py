import numpy as np
import itertools
import random
from hopfield import HopfieldNetwork


def get_initial_patterns():
    """Patrones originales del ejercicio (N, S, C, J)."""
    N = np.array([
        1, -1, -1, -1, 1,
        1, 1, -1, -1, 1,
        1, -1, 1, -1, 1,
        1, -1, -1, 1, 1,
        1, -1, -1, -1, 1
    ])
    S = np.array([
        1, 1, 1, 1, 1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1,
        -1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ])
    C = np.array([
        1, 1, 1, 1, 1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1
    ])
    J = np.array([
        1, 1, 1, 1, 1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        1, -1, 1, -1, -1,
        1, 1, 1, -1, -1
    ])
    return [N, S, C, J], ["N", "S", "C", "J"]


def get_orthogonal_patterns():
    """Patrones diseñados para ser más ortogonales."""
    # 'T' (9 ones)
    T = np.array([
        1, 1, 1, 1, 1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1
    ])
    # 'L' (9 ones)
    L = np.array([
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1
    ])
    # '+' (9 ones)
    PLUS = np.array([
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        1, 1, 1, 1, 1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1
    ])
    return [T, L, PLUS], ["T", "L", "+"]


def get_similar_patterns():
    """Patrones diseñados para ser muy similares (no ortogonales)."""
    # 'C' (13 ones)
    C = np.array([
        1, 1, 1, 1, 1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1
    ])
    # 'O' (16 ones)
    O = np.array([
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ])
    return [C, O], ["C", "O"]


def check_orthogonality(patterns, pattern_names):
    """Calcula el producto punto entre todos los pares de patrones."""
    print("--- Matriz de Ortogonalidad (Producto Punto) ---")
    print(f"Patrones: {', '.join(pattern_names)}")
    print("Diagonal = N (25). Ideal fuera de diagonal = 0.")

    for i, p1_name in enumerate(pattern_names):
        row = f"{p1_name: >4} |"
        for j, p2_name in enumerate(pattern_names):
            dot_product = np.dot(patterns[i], patterns[j])
            row += f" {dot_product/25: .2f} "
        print(row)
    print("-" * 30)


def generate_noisy_pattern(pattern, noise_bits):
    """Invierte 'noise_bits' aleatorios en un patrón."""
    noisy_pattern = pattern.copy()
    noise_idx = np.random.choice(len(pattern), size=noise_bits, replace=False)
    noisy_pattern[noise_idx] *= -1
    return noisy_pattern


def test_stability_and_recall(network, base_patterns, pattern_names):
    """
    Prueba la estabilidad (recall de patrones perfectos) y el recall
    de patrones ruidosos, identificando estados espurios.
    """
    N = network.network_size
    stored_patterns = base_patterns + [p * -1 for p in base_patterns]

    print("\n--- 1. Prueba de Estabilidad (Ruido = 0) ---")
    stable_count = 0
    for i, p in enumerate(base_patterns):
        recalled = network.recall(p)
        is_stable = np.array_equal(recalled, p)
        if is_stable:
            stable_count += 1
        print(f"  Patrón {pattern_names[i]}: {'ESTABLE' if is_stable else 'INESTABLE'}")
    print(f"Resultado: {stable_count}/{len(base_patterns)} patrones son estables.\n")

    print("--- 2. Prueba de Recall vs. Estados Espurios ---")
    noise_levels = [1, 2, 3, 5, 8]
    num_tests_per_level = 100

    print(f"Niveles de ruido (bits): {noise_levels}")
    print(f"Pruebas por nivel/patrón: {num_tests_per_level}\n")
    print(" " * 12 + "| Recall | Inverso | Espurio |")
    print("-" * 43)

    for noise_bits in noise_levels:
        results = {"correct": 0, "inverse": 0, "spurious": 0}

        for base_idx, base_p in enumerate(base_patterns):
            for _ in range(num_tests_per_level):
                noisy_input = generate_noisy_pattern(base_p, noise_bits)

                recalled = network.recall(noisy_input)

                if np.array_equal(recalled, base_p):
                    results["correct"] += 1
                elif np.array_equal(recalled, -base_p):
                    results["inverse"] += 1
                else:
                    is_other_pattern = False
                    for i, p in enumerate(stored_patterns):
                        if i != base_idx and np.array_equal(recalled, p):
                            is_other_pattern = True
                            break

                    if not is_other_pattern:
                        results["spurious"] += 1

                    else:
                        results["spurious"] += 1

        total_tests = len(base_patterns) * num_tests_per_level
        acc_pct = (results["correct"] / total_tests) * 100
        inv_pct = (results["inverse"] / total_tests) * 100
        spu_pct = (results["spurious"] / total_tests) * 100

        print(f"Ruido={noise_bits: 2d} bits | {acc_pct: 6.1f}% | {inv_pct: 6.1f}% | {spu_pct: 6.1f}% |")

def run_analysis(patterns, pattern_names):
    """Ejecuta el análisis completo para un conjunto de patrones."""

    check_orthogonality(patterns, pattern_names)

    hop = HopfieldNetwork(NETWORK_SIZE=25, starting_patterns=patterns)

    test_stability_and_recall(hop, patterns, pattern_names)


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Análisis 1: Patrones Originales (N, S, C, J)")
    print("=" * 50)
    patterns_njsc, names_njsc = get_initial_patterns()
    run_analysis(patterns_njsc, names_njsc)

    print("\n" + "=" * 50)
    print("  Análisis 2: Patrones Ortogonales (T, L, +)")
    print("=" * 50)
    patterns_ortho, names_ortho = get_orthogonal_patterns()
    run_analysis(patterns_ortho, names_ortho)

    print("\n" + "=" * 50)
    print("  Análisis 3: Patrones Similares (C, O)")
    print("=" * 50)
    patterns_sim, names_sim = get_similar_patterns()
    run_analysis(patterns_sim, names_sim)
