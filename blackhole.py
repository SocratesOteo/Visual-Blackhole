import numpy as np
from PIL import Image

# ---  parameters (in units where Rs = 1) ---

W, H = 800, 450    # image size
Rs = 1.0           # Schwarzschild radius
CAPTURE = 2.6 * Rs  #photon sphere (approx) -> black
DISK_IN, DISK_OUT = 3.0, 12.0 # accretion disk extant
FOV = 1.2          # field of view scale


def normalize(v):
    return v / np.linalg.norm(v, axis =-1, keepdims=True)


# --- build a ray direction for every pixel (vectorized) ---

xs = (np.arange(W)- W/2)