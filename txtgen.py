#Este codigo fue creado por IA para fines practicos, es un generador de nodos para las pruebas del proyecto real.

import numpy as np
import matplotlib.image as mpimg
import random
from math import sqrt

# Load image
img = mpimg.imread("image.png")
if len(img.shape) == 3:
    img_gray = img.mean(axis=2)
else:
    img_gray = img

threshold = 0.7
mask = img_gray > threshold
ys, xs = np.where(mask)

H, W = img_gray.shape

# Parameters
N = 20   #Aqui se cambia para mas o menos nodos, con mas de 30 empieza a tardar mucho.
K = 5
RADIUS = 15.0

# Weighted sampling: denser in center
cx, cy = np.mean(xs), np.mean(ys)
dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
dist_norm = dist / dist.max()
weights = (1 - dist_norm)**3
weights /= weights.sum()

# Sample node positions
chosen = np.random.choice(len(xs), size=N, replace=False, p=weights)
positions = [(xs[i], ys[i]) for i in chosen]

# Generate population and critical flags
population = [random.randint(50, 5000) for _ in range(N)]
critical = [1 if random.random() < 0.1 else 0 for _ in range(N)]

# Generate edges using proximity (nearest neighbors graph)
edges = []
for i in range(N):
    x1, y1 = positions[i]
    # distances to all nodes
    dists = []
    for j in range(N):
        if i != j:
            x2, y2 = positions[j]
            d = sqrt((x1 - x2)**2 + (y1 - y2)**2)
            dists.append((d, j))
    dists.sort()
    
    # connect to k nearest neighbors
    for _, j in dists[:5]:
        w = round(random.random() * 10 + 1, 2)
        edges.append((i, j, w))

# Remove duplicates
edges_unique = []
seen = set()
for u,v,w in edges:
    if (v,u) not in seen:
        seen.add((u,v))
        edges_unique.append((u,v,w))

M = len(edges_unique)

# Save dataset
path = "city_dataset.txt"
with open(path, "w") as f:
    f.write(f"{N} {K} {RADIUS}\n")
    for i in range(N):
        f.write(f"{population[i]} {critical[i]}\n")
    f.write(f"{M}\n")
    for u,v,w in edges_unique:
        f.write(f"{u} {v} {w}\n")

path
