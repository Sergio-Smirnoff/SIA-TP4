import numpy as np
import pygame
import os
from hopfield import HopfieldNetwork
from draw_utils import save_gif_5x5, save_matrix_image

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

def accuracy_by_step(hop: HopfieldNetwork):
    patterns = get_initial_patterns()

    #promedio de 10 tiradas para cada patron 
    mean_outputs = [] #-> en cada indice el resultado del promedio para cada patron

    for pattern in patterns:
        
        for i in range(10):
            outputs = [] # -> se va guardando el promedio de la tirada actual para guardarla despues 
            output = hop.recall(pattern)

            #calcular accuracy o recal (alguna metrica)
            #almacenar por patron en la lista de outputs
            print(f"Input    :", pattern)
            print(f"Output   :", output)
            print("——————")




ROWS = 5
COLS = 5

def main():
    # Patrones de entrenamiento (5×5 -> vectores de tamaño 25)
    
    patterns = get_initial_patterns()
    

    # Generar varios inputs ruidosos a partir de patrones
    inputs = []  # incluir patrones originales

    for idx, base in enumerate(patterns):
        for noise_bits in range(25):
            input_vec = base.copy()
            noise_idx = np.random.choice(len(input_vec), size=noise_bits, replace=False)
            input_vec[noise_idx] *= -1
            inputs.append((idx, noise_bits, input_vec))

    # for idx, noise_bits, input in inputs:
        #reset hopfield network for each input
    hop = HopfieldNetwork(NETWORK_SIZE=ROWS*COLS, starting_patterns=patterns)
    ELEMENT = 20
    results_by_step = hop.recall(inputs[ELEMENT][2])  #ejemplo con el input con 2 bits de ruido del patron 0
    print(f'Results for pattern {inputs[ELEMENT][0]} with {inputs[ELEMENT][1]} noise bits:')
    print(results_by_step )
    for step, result in enumerate(results_by_step):
        out_path = f"output_hopfield/pattern_{inputs[ELEMENT][0]}/_noise_{inputs[ELEMENT][1]}/_step_{step}.png"
        save_matrix_image(
            result,
            out_path,
            cell_size=40,
            margin=2,
            on_color=(255, 255, 255),
            off_color=(0, 0, 0),
            bg=(24, 24, 24),
        )

    all_on = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    save_matrix_image(
        all_on,
        "output_hopfield/all_on.png",
        cell_size=40,
        margin=2,
        on_color=(255, 255, 255),
        off_color=(0, 0, 0),
        bg=(24, 24, 24),
    )

    #======== Para comprobar los patrones originales ========
    # a= 22
    # for pattern in patterns:
    #     print(f"Procesando {pattern}")
    #     process_and_save(pattern, hop, idx=(a:=a+1))

    # #se observa un estado espuereo para todos los outputs con la letra C

    # # Esperar para que puedas ver la última pantalla antes de cerrar
    # done = False
    # while not done:
    #     for e in pygame.event.get():
    #         if e.type == pygame.QUIT:
    #             done = True
    # pygame.quit()

    # print("Precisión por paso:")
    # accuracy_by_step(hop)

if __name__ == "__main__":
    main()
