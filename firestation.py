import matplotlib.pyplot as plt
import networkx as nx
import itertools
import math
import time
import random
import matplotlib.image as mpimg
import numpy as np


# DATOS DE LA CIUDAD
def read_city_dataset(path):
    with open(path, "r") as f:
        lines = f.read().strip().split("\n")

    header = lines[0].split()
    N = int(header[0])
    K = int(header[1])
    RADIUS = float(header[2])

    population = []
    critical = []

    idx = 1
    for _ in range(N):
        p, c = lines[idx].split()
        population.append(float(p))
        critical.append(int(c) == 1)
        idx += 1

    M = int(lines[idx])
    idx += 1

    edges = []
    for _ in range(M):
        u, v, w = lines[idx].split()
        edges.append((int(u), int(v), float(w)))
        idx += 1

    return N, K, RADIUS, population, critical, edges


# DIJKSTRA (NETWORKX)
def fast_dijkstra(G, source):
    return nx.single_source_dijkstra_path_length(G, source, weight='weight')


# EVALUAR UNA CONFIGURACIÓN
def evaluate_config(G, stations, population, critical, RADIUS):
    N = len(population)

    bestDist = np.full(N, 1e12)

    for st in stations:
        d = fast_dijkstra(G, st)
        arr = np.array([d.get(i, 1e12) for i in range(N)])
        bestDist = np.minimum(bestDist, arr)

    totalPop = sum(population)

    coveredPop = sum(
        population[i] for i in range(N) if bestDist[i] <= RADIUS
    )
    avg_time = np.average(bestDist, weights=population)

    coverage = coveredPop / totalPop
    score = coverage * 1000 - avg_time

    return score, coverage, avg_time

# BRANCH & BOUND

def branch_and_bound(G, N, K, population, critical, RADIUS, ordered_nodes):
    best_score = -1e18
    best_combo = None
    nodes_checked = 0

    def BnB(index, chosen):
        nonlocal best_score, best_combo, nodes_checked

        if len(chosen) == K:
            nodes_checked += 1
            score, cov, avg = evaluate_config(G, chosen, population, critical, RADIUS)

            if score > best_score:
                best_score = score
                best_combo = chosen[:]
            return

        if len(chosen) + (len(ordered_nodes) - index) < K:
            return

        for i in range(index, len(ordered_nodes)):
            chosen.append(ordered_nodes[i])
            BnB(i + 1, chosen)
            chosen.pop()

    BnB(0, [])
    return best_score, best_combo, nodes_checked

# GENERAR MAPA FROM IMAGE (REALISTA)

def map_from_contour_realistic(G, image_path, threshold=0.7, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    img = mpimg.imread(image_path)
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    mask = img > threshold
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError("La imagen no tiene zonas blancas suficientes.")

    H, W = img.shape

    cx = np.mean(xs)
    cy = np.mean(ys)

    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    dist_norm = dist / dist.max()
    weights = (1 - dist_norm)**3
    weights /= weights.sum()

    pos = {}
    for node in G.nodes():
        idx = np.random.choice(len(xs), p=weights)
        x, y = xs[idx], ys[idx]

        nx = x / W
        ny = y / H

        nx += np.random.uniform(-0.01, 0.01)
        ny += np.random.uniform(-0.01, 0.01)

        pos[node] = (nx, 1 - ny)

    return pos

# GRAFICAR CIUDAD
def plot_city(G, pos, population, critical, stations, RADIUS, image_path=None):
    plt.figure(figsize=(10, 12))

    # Fondo con contorno
    if image_path:
        img = mpimg.imread(image_path)
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        plt.imshow(img, cmap="gray", extent=[0, 1, 0, 1], alpha=0.9)

    normal_nodes = [n for n in G.nodes() if n not in stations and not critical[n]]
    critical_nodes = [n for n in G.nodes() if critical[n] and n not in stations]

    # Tamaño proporcional a población
    normal_size = [population[n] / 25 for n in normal_nodes]
    critical_size = [population[n] / 25 for n in critical_nodes]

    # Nodos normales (azules)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=normal_nodes,
        node_color="skyblue",
        node_size=normal_size,
        edgecolors="black",
        linewidths=0.7,
        alpha=0.9
    )

    # Nodos críticos (naranja)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=critical_nodes,
        node_color="orange",
        node_size=critical_size,
        edgecolors="black",
        linewidths=0.7,
        alpha=0.9
    )

    # Estaciones (rojo)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=stations,
        node_color="red",
        node_size=400,
        edgecolors="black",
        linewidths=1.3,
        alpha=1.0
    )

    # RADIO VISUAL PROPORCIONAL AL MAPA

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    visual_radius = 0.1 * max(xmax - xmin, ymax - ymin)

    for st in stations:
        cx, cy = pos[st]
        circ = plt.Circle(
            (cx, cy),
            visual_radius,
            color="red",
            alpha=0.18,
            linewidth=1.2,
            fill=True
        )
        plt.gca().add_patch(circ)

    plt.title("Fire Station Deployment – Tamaño por población + cobertura")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# MAIN

if __name__ == "__main__":
    DATASET = "city_dataset.txt"
    IMAGE = "image.png"

    N, K, RADIUS, population, critical, edges = read_city_dataset(DATASET)

    G = nx.Graph()
    for i in range(N):
        G.add_node(i)
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    bet = nx.betweenness_centrality(G)
    clo = nx.closeness_centrality(G)
    cent_score = {n: (bet[n] + clo[n]) / 2 for n in G.nodes()}

    TOP = 120 if N > 200 else N
    ordered_nodes = sorted(G.nodes(), key=lambda n: -cent_score[n])[:TOP]

    print(f"Usando los {TOP} nodos más centrales para Branch & Bound.")

    # BRANCH & BOUND
    t0 = time.time()
    best_score, best_stations, checked = branch_and_bound(
        G, N, K, population, critical, RADIUS, ordered_nodes
    )
    t1 = time.time()

    print("\n========== RESULTADOS ==========")
    print("Mejor score:", best_score)
    print("Mejor configuración:", best_stations)
    print("Combinaciones evaluadas:", checked)
    print("Tiempo total:", round(t1 - t0, 2), "s")

    # MAPA
    pos = map_from_contour_realistic(G, IMAGE)
    plot_city(G, pos, population, critical, best_stations, RADIUS, IMAGE)
