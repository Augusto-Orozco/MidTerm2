# Fire Station Deployment Optimization  
**Proyecto MidTerm – Búsqueda y Optimización con IA**

Este proyecto modela y resuelve el _Problema de Despliegue de Estaciones de Bomberos_ utilizando optimización basada en grafos, heurísticas y técnicas de Branch & Bound.  
Se genera una ciudad sintética a partir del contorno geográfico real de una imagen, produciendo un grafo ponderado por población donde cada nodo representa una zona urbana.

El objetivo es encontrar la ubicación óptima de **K estaciones de bomberos** que maximice la cobertura mientras minimiza el tiempo de respuesta ante emergencias.

---

## **1. Características**

### Generación de Ciudad Basada en Imagen
- Carga una imagen del contorno de la ciudad (`image.png`)
- Muestra puntos únicamente dentro del contorno
- Genera:
  - Posiciones de los nodos
  - Población por nodo
  - Nodos críticos (alta prioridad)
  - Conectividad del grafo (carreteras mediante vecinos más cercanos)

### Motor de Optimización
- **Evaluación basada en Dijkstra** (tiempo de respuesta por nodo)
- **Función de puntuación**:
