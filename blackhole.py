import numpy as np
from PIL import Image

# ---  parameters (in units where Rs = 1) ---

W, H = 800, 450    # image size
Rs = 1.0           # Schwarzschild radius
CAPTURE = 2.6 * Rs  #photon sphere (approx) -> black
DISK_IN, DISK_OUT = 3.0, 12.0 # accretion disk extant
CAM = np.array([0.0, 1.5, -18.0]) # Caamera Position
FOV = 1.2          # field of view scale


def normalize(v):
    return v / np.linalg.norm(v, axis =-1, keepdims=True)


# --- build a ray direction for every pixel (vectorized) ---

xs = (np.arange(W)- W/2) / H * FOV
ys = (np.arange(H) - H/2) / H * FOV
xv, yv = np.meshgrid(xs, -ys)
dirs = np.stack([xv,yv,np.ones_like(xv)], axis=-1)
dirs = normalize(dirs)

origin = np.broadcast_to(CAM, dirs.shape)



# --- Step 2: impact parameter for each ray ---
# b = |origin x dir| (since dir is normalized and BH at origin)
cross = np.cross(origin, dirs)
b = np.linalg.norm(cross, axis= - 1)   #(H,W)

# --- Step 3: Capture Mask ---
captured = b < CAPTURE

# Step 4: bend the ray --- 
#Deflection angle; clamp to avoid blowing up near the edge(radius)
alpha = np.clip(2.0 * Rs / np.maximum(b, CAPTURE), 0, 1.5)

# We need to direction toward the Blackhole near the capture radius or edge
# Let n = unit vector from the closest-approach point towards the blackhole 
# a simple approximation: rotate direcction towards the origin_perp
# where origin_ perp is the component of -origin that perpendicular to the origin or the center

to_bh = -origin
perp = to_bh - np.sum(to_bh*dirs, axis=-1, keepdims=True)* dirs
perp = normalize(perp + 1e-9)

new_dir = normalize(
    np.cos(alpha)[..., None]* dirs + np.sin(alpha)[...,None]* perp
)

# --- Step 5: find where bent ray crosses the  bent ray crosses the disk plane (y = 0)
# Parametrize: point = origin + t * new_dir, solve for y=0
#(for a proper simulation you would march the ray in small steps and re-bend;
#  one bend is the thin lens approximation -- fast and able to read)

t_hit = -origin[..., 1] / (new_dir[..., 1] + 1e-9)
hit_pt = origin + t_hit[..., None] * new_dir
r_hit = np.linalg.norm(hit_pt[...,[0,2]], axis=-1)

on_disk = (t_hit > 0) & (r_hit > DISK_IN) & (r_hit < DISK_OUT) & (~captured)

# --- Color the disk: hot(white) inside, cool (red) outside ---
t_norm = np.clip((r_hit - DISK_IN) / (DISK_OUT - DISK_IN), 0, 1)
disk_r = 1.0
disk_g = 0.4 + 0.6 * (1 - t_norm)
disk_b = 0.1 + 0.5 * (1 - t_norm)**2
brightness = (1 - t_norm)**1.2 # The inner ring has to be brighter

# add swirl pattern
phi = np.arctan2(hit_pt[...,2], hit_pt[..., 0])
swirl = 0.7 + 0.3 * np.sin(phi * 3 + r_hit * 0.8)
brightness *= swirl

# --- Starfield for rays that miss the disk or pull of the Blackhole (deterministic hash) ---

def star_bg(dir_v):
    u = (np.arctan2(dir_v[..., 2], dir_v[..., 0]) / (2*np.pi) + 0.5 )* 600
    v = (np.arcsin(np.clip(dir_v[..., 1], -1, 1)) / np.pi +0.5) * 300
    h = np.sin(np.floor(u)*12.9898 + np.floor(v)*78.233)* 43758.5453
    h = h - np.floor(h)
    stars = (h > 0.995).astype(float)* 1.2
    return stars


bg = star_bg(new_dir)


# --- Compose final image ---
img = np.zeros((H, W, 3))
img[..., 0] = np.where(on_disk, disk_r * brightness, bg)
img[..., 1] = np.where(on_disk, disk_g * brightness, bg)
img[..., 2] = np.where(on_disk, disk_b * brightness, bg)
img[captured] = 0

img = np.clip(img, 0, 1)
Image.fromarray((img*255).astype(np.uint8)).save('blackhole.png')
print('saved blackhole.png')










