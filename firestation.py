import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.image as mpimg
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import os
from functools import lru_cache

# =========================
# DATOS DE LA CIUDAD
# =========================
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

# DIJKSTRA
@lru_cache(maxsize=None)
def dijkstra_cached(G_data, source):
    """G_data = list of edges [(u,v,w),...]"""
    G = nx.Graph()
    for u, v, w in G_data:
        G.add_edge(u, v, weight=w)
    return nx.single_source_dijkstra_path_length(G, source, weight='weight')

def compute_all_shortest_paths(G_data, stations, max_workers=4):
    shortest_paths = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(dijkstra_cached, tuple(G_data), st): st for st in stations}
        for fut in as_completed(futures):
            st = futures[fut]
            shortest_paths[st] = fut.result()
    return shortest_paths

# EVALUAR CONFIGURACIÓN
def evaluate_config_cached(shortest_paths, stations, population, RADIUS):
    N = len(population)
    bestDist = np.full(N, 1e12)
    for st in stations:
        d = shortest_paths[st]
        arr = np.array([d.get(i, 1e12) for i in range(N)])
        bestDist = np.minimum(bestDist, arr)
    totalPop = sum(population)
    coveredPop = sum(population[i] for i in range(N) if bestDist[i] <= RADIUS)
    avg_time = np.average(bestDist, weights=population)
    coverage = coveredPop / totalPop
    score = coverage * 1000 - avg_time
    return score, coverage, avg_time

# BRANCH & BOUND
def BnB_global(args):
    """
    Función global para branch & bound, compatible con ProcessPoolExecutor.
    args: (start_idx, chosen_list, ordered_nodes, K, shortest_paths, population, RADIUS, best_score_global)
    """
    start_idx, chosen, ordered_nodes, K, shortest_paths, population, RADIUS, best_score_global = args
    best_score = best_score_global
    best_combo = None
    nodes_checked = 0
    branches_pruned = 0

    stack = [(start_idx, chosen[:])]
    while stack:
        idx, current = stack.pop()
        if len(current) == K:
            nodes_checked += 1
            score, _, _ = evaluate_config_cached(shortest_paths, current, population, RADIUS)
            if score > best_score:
                best_score = score
                best_combo = current[:]
            continue

        if len(current) + (len(ordered_nodes) - idx) < K:
            branches_pruned += 1
            continue

        for i in range(idx, len(ordered_nodes)):
            next_choice = current + [ordered_nodes[i]]
            remaining = K - len(next_choice)
            est_score = len(next_choice) * 1000
            if est_score + remaining * 500 < best_score:
                branches_pruned += 1
                continue
            stack.append((i + 1, next_choice))

    return best_score, best_combo, nodes_checked, branches_pruned

def branch_and_bound_parallel_fixed(G, N, K, population, critical, RADIUS, ordered_nodes, max_workers=4):
    G_data = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
    shortest_paths = {st: dijkstra_cached(tuple(G_data), st) for st in ordered_nodes}
    best_score_global = -1e18

    args_list = [(i, [], ordered_nodes, K, shortest_paths, population, RADIUS, best_score_global)
                 for i in range(len(ordered_nodes))]

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(BnB_global, args) for args in args_list]
        for fut in as_completed(futures):
            results.append(fut.result())

    best_score = -1e18
    best_combo = None
    total_checked = 0
    total_pruned = 0
    for score, combo, checked, pruned in results:
        total_checked += checked
        total_pruned += pruned
        if score > best_score:
            best_score = score
            best_combo = combo

    mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    return best_score, best_combo, total_checked, total_pruned, mem_usage

# MAPA REALISTA
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
    cx, cy = np.mean(xs), np.mean(ys)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    dist_norm = dist / dist.max()
    weights = (1 - dist_norm) ** 3
    weights /= weights.sum()

    pos = {}
    for node in G.nodes():
        idx = np.random.choice(len(xs), p=weights)
        x, y = xs[idx], ys[idx]
        nx_pos = x / W + np.random.uniform(-0.01, 0.01)
        ny_pos = y / H + np.random.uniform(-0.01, 0.01)
        pos[node] = (nx_pos, 1 - ny_pos)
    return pos

# GRAFICAR CIUDAD
def plot_city(G, pos, population, critical, stations, RADIUS, image_path=None):
    plt.figure(figsize=(10, 12))
    if image_path:
        img = mpimg.imread(image_path)
        if len(img.shape) == 3:
            img = img.mean(axis=2)
        plt.imshow(img, cmap="gray", extent=[0, 1, 0, 1], alpha=0.9)

    normal_nodes = [n for n in G.nodes() if n not in stations and not critical[n]]
    critical_nodes = [n for n in G.nodes() if critical[n] and n not in stations]

    nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_color="skyblue",
                           node_size=[population[n]/25 for n in normal_nodes],
                           edgecolors="black", linewidths=0.7, alpha=0.9)

    nx.draw_networkx_nodes(G, pos, nodelist=critical_nodes, node_color="orange",
                           node_size=[population[n]/25 for n in critical_nodes],
                           edgecolors="black", linewidths=0.7, alpha=0.9)

    nx.draw_networkx_nodes(G, pos, nodelist=stations, node_color="red",
                           node_size=400, edgecolors="black", linewidths=1.3, alpha=1.0)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    visual_radius = 0.1 * max(max(xs)-min(xs), max(ys)-min(ys))
    for st in stations:
        cx, cy = pos[st]
        circ = plt.Circle((cx, cy), visual_radius, color="red", alpha=0.18, linewidth=1.2, fill=True)
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

    t0 = time.time()
    best_score, best_stations, checked, pruned, mem_usage = branch_and_bound_parallel_fixed(
        G, N, K, population, critical, RADIUS, ordered_nodes, max_workers=4
    )
    t1 = time.time()

    print("\n========== RESULTADOS ==========")
    print("Mejor score:", best_score)
    print("Mejor configuración:", best_stations)
    print("Combinaciones evaluadas:", checked)
    print("Ramas podadas:", pruned)
    print("Memoria aproximada usada:", round(mem_usage, 2), "MB")
    print("Tiempo total BnB paralelo:", round(t1 - t0, 2), "s")

    pos = map_from_contour_realistic(G, IMAGE)
    plot_city(G, pos, population, critical, best_stations, RADIUS, IMAGE)
