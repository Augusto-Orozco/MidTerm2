# Fire Station Deployment Optimization  
**MidTerm Project – AI Search & Optimization**

This project models and solves the _Fire Station Deployment Problem_ using graph-based optimization, heuristics, and Branch & Bound techniques.  
A synthetic city is generated from a real geographic contour image, producing a population-weighted graph where each node represents an urban zone.

The goal is to find the optimal placement of **K fire stations** that maximizes coverage while minimizing emergency response time.

---

## **1. Features**

### Image-Based City Generation
- Loads a city outline image (`image.png`)
- Samples points only inside the contour
- Generates:
  - Node positions
  - Population per node
  - Critical nodes (high priority)
  - Graph connectivity (nearest-neighbor roads)

### Optimization Engine
- **Dijkstra-based evaluation** (response time per node)
- **Scoring function**:
