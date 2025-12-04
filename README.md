# Optimización de ubicaciones de estaciones de bomberos
### Midterm 2 – TC2038

Este proyecto desarrolla un sistema inteligente para optimizar la ubicación de estaciones de bomberos en una ciudad.  
El enfoque combina **CSP**, **Optimización**, **Grafos**, **Heurísticas de Centralidad**, **Branch & Bound**,  
y un modelo de la ciudad generado desde un **mapa real**.
---

## Problema y Motivación

La correcta ubicación de estaciones de bomberos crítico para la seguridad urbana.

El objetivo es determinar **K ubicaciones óptimas** que:
- Maximizen la cobertura de población  
- Minimicen tiempos de respuesta  
- Atiendan zonas críticas con mayor prioridad  
- Se adapten a la distribución geográfica real de la ciudad  

---

## Objetivo General

Diseñar un sistema que encuentre la **mejor combinación de estaciones** usando:
- Complejidad controlada mediante heurísticas
- Evaluación formal de score, cobertura y tiempo promedio
- Métricas reproducibles para experimentos

---

## Técnica Algorítmica Central

El sistema integra 3 técnicas principales:

### A) **Branch & Bound**
Explora combinaciones posibles sin intentar las que no pueden superar el mejor score parcial.

### B) **Heurísticas de Centralidad**
Ordenamos nodos según:

- **Betweeness Centrality**
- **Closeness Centrality**

Se reduce dramáticamente el espacio de búsqueda manteniendo calidad.

### C) **Geometría Computacional**
El mapa de la ciudad se genera a partir de una imagen real:
- Se detectan contornos  
- Se muestrean coordenadas con densidad variable  
- Las ubicaciones se normalizan a un sistema [0–1]

---

## Arquitectura del Sistema

Dataset (25 nodos)

↓

Graph Builder (NetworkX)

↓

Centrality Ordering

↓

Branch & Bound Core

↓

Evaluator (coverage, score, avg-time)

↓

Visualizer (mapa real con estaciones + círculos proporcionales)

---

## Métricas

Se calcula:

- **Score global = coverage * 1000 – avg_time**
- **Cobertura total**
- **Promedio ponderado de distancia**
- **Combinaciones evaluadas**
- **Tiempo de ejecución**

Para N=25 nodos y K=3, Branch & Bound puede explorar ~200–300 combinaciones.

---

## Datos de Entrada

Archivo `.txt` con:

N K RADIUS

[población] [critical]

M

u v w

u v w

---

## Resultados Visuales

El sistema genera:

- un mapa urbano basado en contorno real  
- colores por tipo de nodo  
- estaciones optimizadas  
- círculos de cobertura **proporcionales al tamaño del mapa**

---

## Requisitos Técnicos

- Python 3  
- NetworkX  
- Matplotlib  
- NumPy  

---

## © Autores
Equipo – TC2038 

José Augusto Orozco Blas
Dario Puentes Diaz
Patricio Flores
Juan Pablo Aguilar Varela
