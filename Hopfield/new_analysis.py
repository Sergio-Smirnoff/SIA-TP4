import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from hopfield import HopfieldNetwork
from draw_utils import save_matrix_image

# ==============================================================
# === Definición de patrones ==================================
# ==============================================================

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
    T = np.array([
        1, 1, 1, 1, 1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1,
        -1, -1, 1, -1, -1
    ])
    L = np.array([
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1
    ])
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
    C = np.array([
        1, 1, 1, 1, 1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, -1, -1, -1, -1,
        1, 1, 1, 1, 1
    ])
    O = np.array([
        1, 1, 1, 1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, -1, -1, -1, 1,
        1, 1, 1, 1, 1
    ])
    return [C, O], ["C", "O"]


# ==============================================================
# === Análisis de ortogonalidad ================================
# ==============================================================

def check_orthogonality(patterns, pattern_names):
    """Imprime matriz de productos punto normalizados."""
    print("--- Matriz de Ortogonalidad (Producto Punto) ---")
    print(f"Patrones: {', '.join(pattern_names)}")
    print("Diagonal = N (25). Ideal fuera de diagonal = 0.")
    for i, p1_name in enumerate(pattern_names):
        row = f"{p1_name: >4} |"
        for j, _ in enumerate(pattern_names):
            dot_product = np.dot(patterns[i], patterns[j])
            row += f" {dot_product/25: .2f} "
        print(row)
    print("-" * 30)


def plot_orthogonality_matrix(patterns, pattern_names):
    """Heatmap de ortogonalidad."""
    n = len(patterns)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = np.dot(patterns[i], patterns[j]) / len(patterns[i])

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=pattern_names, yticklabels=pattern_names, cbar=True)
    plt.title("Matriz de Ortogonalidad Normalizada")
    plt.xlabel("Patrón j")
    plt.ylabel("Patrón i")
    plt.tight_layout()
    plt.show()


# ==============================================================
# === Generación de ruido =====================================
# ==============================================================

def generate_noisy_pattern(pattern, noise_bits):
    """Invierte 'noise_bits' aleatorios en un patrón."""
    noisy_pattern = pattern.copy()
    idx = np.random.choice(len(pattern), size=noise_bits, replace=False)
    noisy_pattern[idx] *= -1
    return noisy_pattern


# ==============================================================
# === Pruebas de estabilidad y recall ==========================
# ==============================================================

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
        recalled = network.recall(p)[-1]
        is_stable = np.array_equal(recalled, p)
        if is_stable:
            stable_count += 1
        print(f"  Patrón {pattern_names[i]}: {'ESTABLE' if is_stable else 'INESTABLE'}")
    print(f"Resultado: {stable_count}/{len(base_patterns)} patrones son estables.\n")

    print("--- 2. Prueba de Recall vs. Estados Espurios ---")
    noise_levels = [1, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25]
    num_tests_per_level = 100
    print(f"Niveles de ruido (bits): {noise_levels}")
    print(f"Pruebas por nivel/patrón: {num_tests_per_level}\n")
    print(" " * 12 + "| Acierto | Inverso | Espurio |")
    print("-" * 43)

    spurious_counts = {}
    inverse_counts = {}

    for noise_bits in noise_levels:
        results = {"correct": 0, "inverse": 0, "spurious": 0}

        for base_idx, base_p in enumerate(base_patterns):
            for _ in range(num_tests_per_level):
                # Reinstancia red (Hopfield con mismos patrones)
                network = HopfieldNetwork(NETWORK_SIZE=25, starting_patterns=base_patterns)
                noisy_input = generate_noisy_pattern(base_p, noise_bits)
                recalled = network.recall(noisy_input)[-1]

                if np.array_equal(recalled, base_p):
                    results["correct"] += 1
                elif np.array_equal(recalled, -base_p):
                    results["inverse"] += 1
                else:
                    # Estado espurio = distinto a cualquier patrón almacenado
                    is_other_pattern = any(np.array_equal(recalled, p)
                                           for i, p in enumerate(stored_patterns)
                                           if i != base_idx)
                    if not is_other_pattern:
                        results["spurious"] += 1
                        if(len(base_patterns) <3):
                            save_matrix_image(recalled, f"spurious_patterns:{len(base_patterns)}_{base_idx}_{noise_bits}.png")
                    else:
                        results["spurious"] += 1
                        if(len(base_patterns) <3):
                            save_matrix_image(recalled, f"spurious_patterns:{len(base_patterns)}_{base_idx}_{noise_bits}.png")

        total_tests = len(base_patterns) * num_tests_per_level
        acc_pct = (results["correct"] / total_tests) * 100
        inv_pct = (results["inverse"] / total_tests) * 100
        spu_pct = (results["spurious"] / total_tests) * 100
        spurious_counts[noise_bits] = spu_pct
        inverse_counts[noise_bits] = inv_pct

        print(f"Ruido={noise_bits: 2d} bits | {acc_pct: 6.1f}% | {inv_pct: 6.1f}% | {spu_pct: 6.1f}% |")

    plot_histograms(spurious_counts, inverse_counts)


# ==============================================================
# === Visualización de resultados =============================
# ==============================================================

def plot_histograms(spurious_counts, inverse_counts):
    """Genera histogramas agrupados para estados espurios e inversos."""
    df_spu = pd.DataFrame({
        "Ruido (bits)": list(spurious_counts.keys()),
        "Espurios (%)": list(spurious_counts.values())
    })
    df_inv = pd.DataFrame({
        "Ruido (bits)": list(inverse_counts.keys()),
        "Inversos (%)": list(inverse_counts.values())
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x="Ruido (bits)", y="Espurios (%)", data=df_spu, ax=axes[0], color="salmon")
    axes[0].set_title("Porcentaje de Estados Espurios vs Ruido")
    axes[0].set_ylabel("Porcentaje")
    axes[0].set_xlabel("Bits alterados")

    sns.barplot(x="Ruido (bits)", y="Inversos (%)", data=df_inv, ax=axes[1], color="skyblue")
    axes[1].set_title("Porcentaje de Estados Inversos vs Ruido")
    axes[1].set_ylabel("Porcentaje")
    axes[1].set_xlabel("Bits alterados")

    plt.tight_layout()
    plt.show()


# ==============================================================
# === Ejecución general del análisis ==========================
# ==============================================================

def run_analysis(patterns, pattern_names):
    """Ejecuta el análisis completo para un conjunto de patrones."""
    check_orthogonality(patterns, pattern_names)
    plot_orthogonality_matrix(patterns, pattern_names)
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