import numpy as np
import pygame
import os
from hopfield import HopfieldNetwork

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

# Configuración
CELL_SIZE = 40
MARGIN = 5
ROWS, COLS = 5, 5
WINDOW_WIDTH  = 600
WINDOW_HEIGHT = 600  # espacio extra para texto

WHITE = (255,255,255)
BLACK = (0,0,0)
GREEN = (0,255,0)
RED   = (255,0,0)
GRAY  = (200,200,200)

pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Hopfield 5×5: Input vs Output")
font = pygame.font.SysFont(None, 24)

def draw_matrix(mat, top_left, title=""):
    x0, y0 = top_left
    if title:
        text = font.render(title, True, BLACK)
        screen.blit(text, (x0, y0 - 30))
    for i in range(ROWS):
        for j in range(COLS):
            val = mat[i][j]
            color = GREEN if val == 1 else RED
            rect = pygame.Rect(
                x0 + MARGIN + j*(CELL_SIZE+MARGIN),
                y0 + MARGIN + i*(CELL_SIZE+MARGIN),
                CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

def matrix_from_vector(vec):
    return vec.reshape((ROWS, COLS))

def save_surface(filename):
    """Guarda la pantalla actual en archivo."""
    pygame.image.save(screen, filename)
    print(f"Guardado: {filename}")

def process_and_save(input_vec, hop, idx, output_dir="results"):
    output_vec = hop.recall(input_vec)
    mat_in  = matrix_from_vector(input_vec)
    mat_out = matrix_from_vector(output_vec)

    # Dibujo
    screen.fill(WHITE)
    draw_matrix(mat_in,  (10,10),   title="Input #%d" % idx)
    draw_matrix(mat_out, (10,10 + ROWS*(CELL_SIZE+MARGIN) + 40), title="Output #%d" % idx)
    pygame.display.flip()

    # Crear carpeta si no existe
    os.makedirs(output_dir, exist_ok=True)
    # Guardar
    fname = os.path.join(output_dir, f"hopfield_pair_{idx:03d}.png")
    save_surface(fname)

def main():
    # Patrones de entrenamiento (5×5 -> vectores de tamaño 25)
    
    patterns = get_initial_patterns()
    
    hop = HopfieldNetwork(NETWORK_SIZE=ROWS*COLS, starting_patterns=patterns)

    # Generar varios inputs ruidosos a partir de patrones
    inputs = []  # incluir patrones originales

    for idx, base in enumerate(patterns):
        for noise_bits in [1, 2, 3]:
            input_vec = base.copy()
            noise_idx = np.random.choice(len(input_vec), size=noise_bits, replace=False)
            input_vec[noise_idx] *= -1
            inputs.append((idx, noise_bits, input_vec))

    # Procesar cada input e ir guardando
    for i, (base_idx, noise_bits, inp) in enumerate(inputs, start=1):
        print(f"Procesando caso {i}: base patrón {base_idx}, ruido={noise_bits} bits")
        process_and_save(inp, hop, idx=i)
    
    #======== Para comprobar los patrones originales ========
    a= 22
    for pattern in patterns:
        print(f"Procesando {pattern}")
        process_and_save(pattern, hop, idx=(a:=a+1))

    #se observa un estado espuereo para todos los outputs con la letra C

    # Esperar para que puedas ver la última pantalla antes de cerrar
    done = False
    while not done:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                done = True
    pygame.quit()

if __name__ == "__main__":
    main()
