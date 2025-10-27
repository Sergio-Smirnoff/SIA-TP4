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

def process_and_save(input_vec, hop: HopfieldNetwork, idx, output_dir="results"):
    output_vec = hop.recall(input_vec)[-1] # obtener el último estado
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


# -*- coding: utf-8 -*-
"""
Pygame viewer for two matrices side-by-side, and GIF saver for frames of matrices.

Requirements:
  pip install pygame pillow

Usage:
  - Run to see the demo.
  - Press ESC or close the window to exit the viewer.
"""

from typing import List, Optional, Tuple
import pygame
from PIL import Image
import math

Matrix = List[List[float]]

# -----------------------------
# Utility: grayscale mapping
# -----------------------------
def compute_global_min_max(mats: List[Matrix]) -> Tuple[float, float]:
    vmin = math.inf
    vmax = -math.inf
    for M in mats:
        for row in M:
            for v in row:
                if v < vmin: vmin = v
                if v > vmax: vmax = v
    if vmin == math.inf:  # empty safeguard
        vmin, vmax = 0.0, 1.0
    if math.isclose(vmin, vmax):
        # avoid divide-by-zero; spread slightly
        vmin, vmax = vmin - 0.5, vmax + 0.5
    return vmin, vmax


def matrix_to_surface(
    M: Matrix,
    cell_size: int = 20,
    margin: int = 1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    bg: Tuple[int, int, int] = (24, 24, 24)
) -> pygame.Surface:
    """
    Convert a numeric matrix to a Pygame Surface using grayscale cells.
    - M: list of lists (rows) with numeric values.
    - cell_size: pixel size of each cell.
    - margin: inner spacing between cells (0..).
    - vmin/vmax: normalization range; if None, computed from M.

    Returns: pygame.Surface (no display needed).
    """
    rows = len(M)
    cols = len(M[0]) if rows > 0 else 0

    if rows == 0 or cols == 0:
        # Return a tiny surface if empty
        surf = pygame.Surface((1, 1), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        return surf

    if vmin is None or vmax is None:
        vmin, vmax = compute_global_min_max([M])

    w = cols * cell_size
    h = rows * cell_size
    surf = pygame.Surface((w, h))
    surf.fill(bg)

    # Draw cells
    for r in range(rows):
        for c in range(cols):
            v = M[r][c]
            t = (v - vmin) / (vmax - vmin) if not math.isclose(vmin, vmax) else 0.5
            t = 0.0 if t < 0 else 1.0 if t > 1 else t
            gray = int(round(255 * t))
            color = (gray, gray, gray)
            x = c * cell_size
            y = r * cell_size
            rect = pygame.Rect(x, y, cell_size, cell_size)
            if margin > 0:
                inner = rect.inflate(-margin, -margin)
                pygame.draw.rect(surf, color, inner)
            else:
                pygame.draw.rect(surf, color, rect)

    return surf


# -----------------------------
# Part 1: Draw two matrices
# -----------------------------
def draw_two_matrices(
    A: Matrix,
    B: Matrix,
    cell_size: int = 20,
    margin: int = 1,
    gap: int = 20,
    window_title: str = "Two Matrices Viewer",
    bg: Tuple[int, int, int] = (18, 18, 18),
    same_scale: bool = True,
) -> None:
    """
    Opens a Pygame window and draws A and B side-by-side.
    - same_scale=True: normalizes both matrices with the same global vmin/vmax for comparable contrast.
    """
    pygame.init()
    pygame.display.set_caption(window_title)

    # Compute normalization
    vmin = vmax = None
    if same_scale:
        vmin, vmax = compute_global_min_max([A, B])

    surf_A = matrix_to_surface(A, cell_size=cell_size, margin=margin, vmin=vmin, vmax=vmax, bg=bg)
    surf_B = matrix_to_surface(B, cell_size=cell_size, margin=margin, vmin=vmin, vmax=vmax, bg=bg)

    # Layout
    h = max(surf_A.get_height(), surf_B.get_height())
    w = surf_A.get_width() + gap + surf_B.get_width()
    screen = pygame.display.set_mode((w, h))
    screen.fill(bg)

    # Center vertically if heights differ
    yA = (h - surf_A.get_height()) // 2
    yB = (h - surf_B.get_height()) // 2

    # Main loop
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill(bg)
        screen.blit(surf_A, (0, yA))
        screen.blit(surf_B, (surf_A.get_width() + gap, yB))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


# -----------------------------
# Part 2: Save GIF from frames
# -----------------------------
def surface_to_pil_rgba(surf: pygame.Surface) -> Image.Image:
    """Convert a Pygame Surface (RGBA) to a PIL Image."""
    w, h = surf.get_size()
    data = pygame.image.tostring(surf, "RGBA")
    im = Image.frombytes("RGBA", (w, h), data)
    return im


def save_gif(
    frames: List[Matrix],
    out_path: str,
    cell_size: int = 20,
    margin: int = 1,
    bg: Tuple[int, int, int] = (18, 18, 18),
    duration_ms: int = 120,
    loop: int = 0,
    same_scale: bool = True,
) -> None:
    """
    Save an animated GIF from a list of matrices (one matrix per frame).
    - frames: list of matrices (list of lists) of equal shape (recommended).
    - same_scale=True: uses a global vmin/vmax computed across all frames (keeps contrast consistent).
    """
    if len(frames) == 0:
        raise ValueError("frames is empty.")

    # Compute global vmin/vmax for consistent scaling across frames
    vmin = vmax = None
    if same_scale:
        vmin, vmax = compute_global_min_max(frames)

    # Render each frame to a Surface, then to PIL
    pil_frames = []
    for M in frames:
        surf = matrix_to_surface(M, cell_size=cell_size, margin=margin, vmin=vmin, vmax=vmax, bg=bg)
        im_rgba = surface_to_pil_rgba(surf)
        # Convert to palette-based for GIF; 'adaptive' preserves detail
        im_p = im_rgba.convert("P", palette=Image.ADAPTIVE)
        pil_frames.append(im_p)

    # Save GIF
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=2,
    )


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

    print("Precisión por paso:")
    accuracy_by_step(hop)

if __name__ == "__main__":
    main()
