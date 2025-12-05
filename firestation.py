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
import math

# 1. LECTURA DE DATOS
def read_city_dataset(path):
    if not os.path.exists(path):
        # Generación de datos sintéticos
        N, K, R = 60, 3, 0.2
        pop = [random.uniform(10, 100) for _ in range(N)]
        crit = [random.random() < 0.2 for _ in range(N)]
        edges = []
        for i in range(N):
            for j in range(i+1, N):
                if random.random() < 0.15: 
                    w = random.uniform(0.05, 0.3)
                    edges.append((i, j, w))
        return N, K, R, pop, crit, edges

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

# 2. MOTORES DE CÁLCULO
@lru_cache(maxsize=None)
def dijkstra_cached(G_data, source):
    G = nx.Graph()
    for u, v, w in G_data:
        G.add_edge(u, v, weight=w)
    return nx.single_source_dijkstra_path_length(G, source, weight='weight')

def evaluate_config_fast(shortest_paths, stations, population, RADIUS):
    N = len(population)
    bestDist = np.full(N, 1e12)
    
    valid_stations = [st for st in stations if st in shortest_paths]
    if not valid_stations: return -1e12, 0, 0

    for st in valid_stations:
        d = shortest_paths[st]
        arr = np.array([d.get(i, 1e12) for i in range(N)])
        bestDist = np.minimum(bestDist, arr)
    
    totalPop = sum(population)
    covered_indices = bestDist <= RADIUS
    coveredPop = np.sum(np.array(population)[covered_indices])
    
    if totalPop > 0:
        avg_time = np.average(bestDist, weights=population)
        coverage = coveredPop / totalPop
    else:
        avg_time = 0
        coverage = 0
        
    score = coverage * 1000 - avg_time
    return score, coverage, avg_time

# 3. BRANCH & BOUND
def BnB_global(args):
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
            score, _, _ = evaluate_config_fast(shortest_paths, current, population, RADIUS)
            if score > best_score:
                best_score = score
                best_combo = current[:]
            continue

        if len(current) + (len(ordered_nodes) - idx) < K:
            branches_pruned += 1
            continue

        for i in range(idx, len(ordered_nodes)):
            next_node = ordered_nodes[i]
            next_choice = current + [next_node]
            
            upper_bound = 1000 
            if upper_bound < best_score:
                branches_pruned += 1
                continue
            
            stack.append((i + 1, next_choice))

    return best_score, best_combo, nodes_checked, branches_pruned

def branch_and_bound_parallel_fixed(G, N, K, population, critical, RADIUS, ordered_nodes, max_workers=None, verbose=True):
    if max_workers is None:
        max_workers = os.cpu_count()
    
    if verbose: print(f"   -> Iniciando B&B con {max_workers} workers...")
    
    G_data = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
    
    # Pre-cálculo Dijkstra (secuencial o paralelo según necesidad, aquí optimizado)
    shortest_paths = {}
    # Usamos executor solo si vale la pena, sino secuencial para overhead bajo en benchmarks pequeños
    if len(ordered_nodes) > 10:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(dijkstra_cached, tuple(G_data), st): st for st in ordered_nodes}
            for fut in as_completed(futures):
                st = futures[fut]
                shortest_paths[st] = fut.result()
    else:
        for st in ordered_nodes:
            shortest_paths[st] = dijkstra_cached(tuple(G_data), st)

    best_score_global = -1e18

    args_list = []
    for i in range(len(ordered_nodes)):
        args = (i + 1, [ordered_nodes[i]], ordered_nodes, K, shortest_paths, population, RADIUS, best_score_global)
        args_list.append(args)

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(BnB_global, args) for args in args_list]
        for fut in as_completed(futures):
            results.append(fut.result())

    best_score = -1e18
    best_combo = []
    total_checked = 0
    total_pruned = 0
    
    for score, combo, checked, pruned in results:
        total_checked += checked
        total_pruned += pruned
        if score > best_score and combo is not None:
            best_score = score
            best_combo = combo

    return best_score, best_combo, total_checked, total_pruned, shortest_paths

# 4. VISUALIZACIÓN
def map_from_contour_realistic(G, image_path, threshold=0.7, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    if not image_path or not os.path.exists(image_path):
        return nx.spring_layout(G, seed=seed)

    img = mpimg.imread(image_path)
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    mask = img > threshold 
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return nx.spring_layout(G, seed=seed)

    H, W = img.shape
    
    cx, cy = np.mean(xs), np.mean(ys)
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    max_dist = dist.max() if dist.max() > 0 else 1
    dist_norm = dist / max_dist
    
    weights = (1 - dist_norm) ** 3
    weights_sum = weights.sum()
    if weights_sum == 0:
        weights = np.ones(len(weights)) / len(weights)
    else:
        weights /= weights_sum

    pos = {}
    valid_indices = np.arange(len(xs))
    
    for node in G.nodes():
        idx = np.random.choice(valid_indices, p=weights)
        x, y = xs[idx], ys[idx]
        nx_pos = x / W + np.random.uniform(-0.01, 0.01)
        ny_pos = y / H + np.random.uniform(-0.01, 0.01)
        pos[node] = (nx_pos, 1 - ny_pos)
        
    return pos

def plot_poster_graphs(G, pos, critical, stations, image_path=None):
    # GRAFICA 1: Estructura
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_size=20, node_color="#999999", edge_color="#CCCCCC", width=0.5, with_labels=False)
    crit_nodes = [n for n, c in enumerate(critical) if c]
    nx.draw_networkx_nodes(G, pos, nodelist=crit_nodes, node_color="orange", node_size=50, label="Puntos Críticos")
    plt.axis("off")
    plt.savefig("poster_graph_structure.png", dpi=300, bbox_inches='tight')
    plt.close()

    # GRAFICA 2: Solución Final
    plt.figure(figsize=(10, 10))
    if image_path and os.path.exists(image_path):
        img = mpimg.imread(image_path)
        plt.imshow(img, cmap="gray", extent=[0, 1, 0, 1], alpha=0.3)
    
    nx.draw_networkx_nodes(G, pos, node_size=10, node_color="#AAAAAA", alpha=0.6)
    nx.draw_networkx_nodes(G, pos, nodelist=stations, node_color="#D90429", node_size=300, edgecolors="black", label="Estaciones")
    
    ax = plt.gca()
    for st in stations:
        if st in pos:
            c = plt.Circle(pos[st], 0.15, color="#D90429", alpha=0.2)
            ax.add_patch(c)
    
    plt.title(f"Despliegue Óptimo (N={len(G.nodes())}, K={len(stations)})")
    plt.axis("off")
    plt.savefig("poster_final_map.png", dpi=300, bbox_inches='tight')
    plt.close()

# 5. MÓDULO DE BENCHMARKING (A, B, 2A)
def run_all_benchmarks(G, N, population, critical, RADIUS, ordered_nodes, best_score_bnb, best_stations_bnb, time_bnb):
    print("\n" + "="*40)
    print("   EJECUTANDO BENCHMARKS PAR EL PÓSTER")
    print("="*40)

    # --- DATOS NECESARIOS ---
    G_data = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
    
    # --- BENCHMARK A: Comparativa (Random vs Greedy vs B&B) ---
    print("\n[A] Generando Comparativa de Algoritmos...")
    
    # 1. Random (Promedio de 10)
    rand_scores = []
    rand_covs = []
    for _ in range(10):
        r_st = random.sample(list(G.nodes()), len(best_stations_bnb))
        # Calculamos caminos al vuelo
        paths = {st: dijkstra_cached(tuple(G_data), st) for st in r_st}
        s, c, _ = evaluate_config_fast(paths, r_st, population, RADIUS)
        rand_scores.append(s)
        rand_covs.append(c)
    avg_rand_score = np.mean(rand_scores)
    avg_rand_cov = np.mean(rand_covs)

    # 2. Greedy (Top Centralidad)
    k = len(best_stations_bnb)
    greedy_st = ordered_nodes[:k]
    paths_g = {st: dijkstra_cached(tuple(G_data), st) for st in greedy_st}
    greedy_score, greedy_cov, _ = evaluate_config_fast(paths_g, greedy_st, population, RADIUS)
    
    # 3. Datos B&B (Ya calculados)
    # Recalculamos cov exacto para el plot
    paths_bnb = {st: dijkstra_cached(tuple(G_data), st) for st in best_stations_bnb}
    _, bnb_cov, _ = evaluate_config_fast(paths_bnb, best_stations_bnb, population, RADIUS)

    # Plot Comparativo
    methods = ['Aleatorio', 'Heurística', 'Branch & Bound']
    scores = [avg_rand_score, greedy_score, best_score_bnb]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(methods, scores, color=['#adb5bd', '#495057', '#d90429'])
    plt.ylabel('Score (Cobertura Pond. - Tiempo)')
    plt.title('Comparativa: Calidad de la Solución')
    plt.ylim(0, max(scores)*1.2)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
    plt.savefig("benchmark_comparison.png", dpi=300)
    print("   -> 'benchmark_comparison.png' guardado.")

    print(f"\n   TABLA PARA LATEX (Punto A):")
    print(f"   Random     | Cov: {avg_rand_cov*100:.1f}% | Score: {avg_rand_score:.1f}")
    print(f"   Heuristic  | Cov: {greedy_cov*100:.1f}% | Score: {greedy_score:.1f}")
    print(f"   B&B (Ours) | Cov: {bnb_cov*100:.1f}% | Score: {best_score_bnb:.1f}")


    # --- BENCHMARK 2A: Speedup (Ley de Amdahl) ---
    print("\n[2A] Calculando Speedup (esto puede tardar unos segundos)...")
    cpu_counts = [1, 2, 4]
    if os.cpu_count() >= 8: cpu_counts.append(8)
    
    times = []
    # Usamos una instancia ligeramente reducida para el speedup para no tardar años si N es grande
    K_bench = min(3, len(best_stations_bnb))
    
    for c in cpu_counts:
        t0 = time.time()
        branch_and_bound_parallel_fixed(G, N, K_bench, population, critical, RADIUS, ordered_nodes, max_workers=c, verbose=False)
        dur = time.time() - t0
        times.append(dur)
        print(f"   -> {c} cores: {dur:.2f}s")
        
    plt.figure(figsize=(7, 5))
    plt.plot(cpu_counts, times, marker='o', linestyle='-', color='#003049', linewidth=2, markersize=8)
    plt.xlabel('Número de Workers (Núcleos)')
    plt.ylabel('Tiempo de Ejecución (s)')
    plt.title('Análisis de Speedup (Multiprocessing)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(cpu_counts)
    plt.savefig("benchmark_speedup.png", dpi=300)
    print("   -> 'benchmark_speedup.png' guardado.")


    # --- BENCHMARK B: Eficiencia de Poda ---
    print("\n[B] Analizando Eficiencia de Poda (Variando K)...")
    print(f"{'K':<5} | {'Comb. Teoricas':<15} | {'Nodos Evaluados':<15} | {'% Podado':<10}")
    print("-" * 55)
    
    # Probamos con K=1, 2, 3 (y 4 si es rápido)
    k_values = [1, 2, 3]
    top_n_eff = min(len(ordered_nodes), 50) # Limitamos el espacio de búsqueda para el benchmark
    nodes_eff = ordered_nodes[:top_n_eff]
    
    for k_val in k_values:
        total_combinations = math.comb(top_n_eff, k_val)
        
        _, _, checked, pruned, _ = branch_and_bound_parallel_fixed(
            G, N, k_val, population, critical, RADIUS, nodes_eff, max_workers=4, verbose=False
        )
        
        # El numero real de nodos visitados en el arbol de estado vs combinaciones hojas
        pruned_pct = 100 * (1 - (checked / total_combinations)) if total_combinations > 0 else 0
        if pruned_pct < 0: pruned_pct = 0 # Ajuste por nodos intermedios
        
        print(f"{k_val:<5} | {total_combinations:<15} | {checked:<15} | {pruned_pct:.2f}%")

# 6. MAIN
if __name__ == "__main__":
    DATASET = "city_dataset.txt" 
    IMAGE = "mapa1.png"

    print("--- OPTIMIZACIÓN DE ESTACIONES DE BOMBEROS ---")
    N, K, RADIUS, population, critical, edges = read_city_dataset(DATASET)
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    bet = nx.betweenness_centrality(G)
    clo = nx.closeness_centrality(G)
    cent_score = {n: (bet[n] + clo[n]) / 2 for n in G.nodes()}
    
    TOP_LIMIT = 50 if N > 100 else N
    ordered_nodes = sorted(G.nodes(), key=lambda n: -cent_score[n])[:TOP_LIMIT]

    t0 = time.time()
    best_score, best_stations, checked, pruned, _ = branch_and_bound_parallel_fixed(
        G, N, K, population, critical, RADIUS, ordered_nodes
    )
    t1 = time.time()
    main_time = t1 - t0

    print(f"   Estaciones Óptimas: {best_stations}")
    print(f"   Score: {best_score:.2f}")

    # Visualización Principal
    pos = map_from_contour_realistic(G, IMAGE)
    plot_poster_graphs(G, pos, critical, best_stations, IMAGE)
    
    # --- EJECUTAR BENCHMARKS SOLICITADOS ---
    run_all_benchmarks(G, N, population, critical, RADIUS, ordered_nodes, best_score, best_stations, main_time)
