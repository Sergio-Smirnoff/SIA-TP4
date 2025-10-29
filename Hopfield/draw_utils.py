import os
import math
from typing import List, Sequence, Tuple, Union
import numpy as np
import pygame
from PIL import Image
import matplotlib.pyplot as plt

MatrixLike = Union[List[List[int]], List[int], np.ndarray]

# ----------------------------
# Pygame init (headless-safe)
# ----------------------------
def ensure_pygame():
    # Allow headless runs (no window)
    if "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    if not pygame.get_init():
        pygame.init()

# ----------------------------
# Matrix utilities
# ----------------------------
def to_5x5_binary(M: MatrixLike) -> np.ndarray:
    """Accepts (5,5) or length-25; returns np.int8 matrix with values in {-1, 1}."""
    arr = np.array(M)
    if arr.shape == (25,):
        arr = arr.reshape(5, 5)
    elif arr.shape != (5, 5):
        raise ValueError(f"Expected shape (5,5) or (25,), got {arr.shape}")

    # Normalize values to {-1, 1}
    unique = np.unique(arr)
    if set(unique.tolist()).issubset({-1, 1}):
        return arr.astype(np.int8)
    if set(unique.tolist()).issubset({0, 1}):
        return np.where(arr == 1, 1, -1).astype(np.int8)
    raise ValueError("Matrix must contain only {-1,1} or {0,1} values.")

# ----------------------------
# Drawing
# ----------------------------
def matrix_to_surface(
    M: MatrixLike,
    cell_size: int = 40,
    margin: int = 2,
    on_color: Tuple[int, int, int] = (255, 255, 255),  # for +1
    off_color: Tuple[int, int, int] = (0, 0, 0),       # for -1
    bg: Tuple[int, int, int] = (24, 24, 24),
) -> pygame.Surface:
    """Render a 5x5 binary matrix to a Pygame Surface."""
    ensure_pygame()
    A = to_5x5_binary(M)

    rows, cols = 5, 5
    w = cols * cell_size
    h = rows * cell_size

    surf = pygame.Surface((w, h))
    surf.fill(bg)

    for r in range(rows):
        for c in range(cols):
            val = A[r, c]
            color = on_color if val == 1 else off_color
            x = c * cell_size
            y = r * cell_size
            rect = pygame.Rect(x, y, cell_size, cell_size)
            if margin > 0:
                inner = rect.inflate(-margin, -margin)
                pygame.draw.rect(surf, color, inner)
            else:
                pygame.draw.rect(surf, color, rect)
    return surf

def surface_to_pil_rgba(surf: pygame.Surface) -> Image.Image:
    """Convert Pygame Surface to PIL Image (RGBA)."""
    w, h = surf.get_size()
    raw = pygame.image.tostring(surf, "RGBA")
    return Image.frombytes("RGBA", (w, h), raw)

import os
from PIL import Image
import pygame
import numpy as np

def save_matrix_image(
    M,
    out_path: str,
    cell_size: int = 40,
    margin: int = 2,
    on_color=(255, 255, 255),
    off_color=(0, 0, 0),
    bg=(24, 24, 24)
):
    """
    Render a single 5x5 binary matrix and save it as an image (PNG, JPG...).
    Accepts values in {-1, 1} or {0, 1}.
    """

    # --- normalize values to {-1, 1} ---
    arr = np.array(M)
    if arr.shape == (25,):
        arr = arr.reshape(5, 5)
    elif arr.shape != (5, 5):
        raise ValueError(f"Expected shape (5,5) or (25,), got {arr.shape}")

    if set(np.unique(arr)) == {0, 1}:
        arr = np.where(arr == 1, 1, -1)

    # --- render surface using existing matrix_to_surface ---
    surf = matrix_to_surface(arr, cell_size, margin, on_color, off_color, bg)

    # --- convert to PIL image ---
    w, h = surf.get_size()
    raw = pygame.image.tostring(surf, "RGBA")
    im = Image.frombytes("RGBA", (w, h), raw)

    # --- ensure output directory exists ---
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # --- save image ---
    im.save(out_path)
    print(f"Saved matrix image: {out_path}")


# ----------------------------
# GIF saver
# ----------------------------
def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:  # only if there's a directory component
        os.makedirs(parent, exist_ok=True)

def save_gif_5x5(
    frames: Sequence[MatrixLike],
    out_path: str,
    cell_size: int = 40,
    margin: int = 2,
    on_color: Tuple[int, int, int] = (255, 255, 255),
    off_color: Tuple[int, int, int] = (0, 0, 0),
    bg: Tuple[int, int, int] = (24, 24, 24),
    fps: int = 2,
    loop: int = 0,
) -> None:
    """
    Render each 5x5 frame with pygame and save as an animated GIF.
    - frames: list of matrices (each (5,5) or length-25), binary in {-1,1} or {0,1}
    - fps: frames per second for the GIF
    - loop: 0 = loop forever, N = loop N times
    """
    if len(frames) == 0:
        raise ValueError("frames is empty")

    ensure_pygame()
    _ensure_parent_dir(out_path)
    pil_frames: List[Image.Image] = []
    for M in frames:
        surf = matrix_to_surface(M, cell_size, margin, on_color, off_color, bg)
        pil = surface_to_pil_rgba(surf).convert("P", palette=Image.ADAPTIVE)
        pil_frames.append(pil)

    duration_ms = int(1000 / max(1, fps))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
        disposal=2,
    )

def get_pattern_letter(pattern_idx: int) -> str:
    if(pattern_idx == 0):
       return "N"
    elif(pattern_idx == 1):
       return "S"
    elif(pattern_idx == 2):
       return "C"
    elif(pattern_idx == 3):
       return "J"


def plot_accuracy(noise_bits, accuracy, accuracy_per_pattern=None, title="Tasa de aciertos vs. ruido"):
    """
    Simple matplotlib plotter.
    """
    plt.figure(figsize=(7, 4.5))
    # global curve
    plt.plot(noise_bits, accuracy, label='Media global')
    # optionally show faint per-pattern curves
    if accuracy_per_pattern is not None:
        for i in range(accuracy_per_pattern.shape[0]):
            plt.plot(
                noise_bits, 
                accuracy_per_pattern[i],
                alpha=0.5,
                label=f'{get_pattern_letter(i)}')
            

    plt.xlabel("Bits flippeados")
    plt.ylabel("Tasa de aciertos")
    plt.title(title)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()