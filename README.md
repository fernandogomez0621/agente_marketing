

## Acceso Producción — 3 Puertos

| Puerto | Servicio | URL | Qué es |
|---------|----------|-----|--------|
| **3000** | React Frontend | http://3.16.212.12:3000 | Chat con los agentes, vista de clusters e historial de trazas |
| **8000** | FastAPI API | http://3.16.212.12:8000/docs | Swagger UI con los endpoints REST |
| **5000** | MLflow | http://3.16.212.12:5000 | Tracking de experimentos, métricas y artefactos |

# Segmentación de Clientes — Challenge

Sistema de análisis de segmentación de clientes con **multi-agente conversacional**
basado en LangGraph, trazabilidad con MLflow, y API REST con interfaz React.

**Autor:** Andrés Fernando Gómez Rojas

## Resumen

Pipeline de segmentación no supervisada sobre 9,996 visitantes de un sitio e-commerce
(Google Analytics). La solución usa **KMeans con K=4**, validado por bootstrap (ARI 0.97),
tests estadísticos (Kruskal-Wallis, MANOVA), y análisis de estabilidad.

Sobre la segmentación se construye un **sistema multi-agente** con 3 especialistas
(Analyst, Statistician, Strategist) que colaboran para responder consultas complejas.
El orquestador descompone queries y puede enviar sub-tareas a **varios agentes en paralelo**.

## Quick Start

```bash
# 1. Configurar API key
cp .env.example .env
# Editar .env con tu ANTHROPIC_API_KEY

# 2. Levantar los 3 servicios
docker-compose up --build

# 3. Esperar ~60s la primera vez (instala dependencias)
```

## Acceso — 3 Puertos

| Puerto | Servicio | URL | Qué es |
|--------|----------|-----|--------|
| **3000** | React Frontend | http://localhost:3000 | Chat con los agentes, vista de clusters, historial de traces |
| **8000** | FastAPI API | http://localhost:8000/docs | Swagger UI con los 6 endpoints REST |
| **5000** | MLflow | http://localhost:5000 | Tracking de experimentos, métricas, artefactos |




### Probar con curl (sin frontend)

```bash
# Health check
curl http://localhost:8000/health | python -m json.tool

# Ver clusters
curl http://localhost:8000/clusters | python -m json.tool

# Consulta al sistema multi-agente
curl -X POST http://localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Analiza el cluster 3 y sugiere estrategia de marketing"}'

# Ver traces de MLflow
curl http://localhost:8000/traces | python -m json.tool
```

## Estructura

```
challenge_segmentacion/
├── notebooks/                  # Análisis paso a paso
│   ├── 01_eda.ipynb            # Exploración de datos
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_clustering.ipynb     # KMeans K=4, validación
│   ├── 04_tests_estadisticos.ipynb
│   └── 05_perfilado_recomendaciones.ipynb
├── src/
│   ├── data_loader.py          # Carga y validación
│   ├── preprocessing.py        # Feature engineering
│   ├── clustering.py           # KMeans, benchmark, validación
│   ├── statistical_tests.py    # Kruskal-Wallis, chi², MANOVA
│   ├── agents/                 # Sistema multi-agente
│   │   ├── tools.py            # 9 tools con @tool (5 capas)
│   │   ├── graph.py            # LangGraph StateGraph + MLflow
│   │   └── config.py
│   └── api/
│       └── main.py             # FastAPI (5 endpoints)
├── frontend/
│   └── index.html              # React SPA
├── data/
│   ├── raw/                    # Dataset original
│   └── processed/              # Datos segmentados
├── models/                     # scaler.pkl, kmeans_k4.pkl
├── report/                     # Informe PDF + LaTeX
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Arquitectura del Sistema Multi-Agente

```
Query del usuario
       │
       ▼
  [SUPERVISOR]  ← Clasifica scope, descompone, planifica
       │
       ├── analyst      → get_cluster_profile, compare_clusters,
       │                  rank_cluster_drivers, get_cluster_overview
       │
       ├── statistician → test_significance, count_visitors,
       │                  get_top_categories
       │
       └── strategist   → suggest_marketing_strategy (reglas codificadas)
       │
       ▼
  [SYNTHESIZER] → Combina outputs en respuesta coherente
```

**Diferenciadores:**
- El supervisor puede enviar a **1, 2, o 3 agentes** por consulta
- Fan-out paralelo con LangGraph `StateGraph` + reducers
- Las estrategias de marketing son **pre-codificadas** (no hallucination)
- Queries fuera de scope son rechazadas con fallback amable
- Toda la trazabilidad en **MLflow**: query, agentes, tools, latencia

## API Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Estado del sistema |
| GET | `/clusters` | Overview de los 4 clusters |
| GET | `/clusters/{id}` | Perfil detallado de un cluster |
| GET | `/clusters/{id}/strategy` | Estrategia de marketing |
| POST | `/agent/query` | Consulta al sistema multi-agente |
| GET | `/traces` | Historial de ejecuciones (MLflow) |

## Segmentos Identificados

| Cluster | Nombre | % | Perfil |
|---------|--------|---|--------|
| C0 | Desktop Engaged | 42.5% | Alto engagement, bajo bounce, desktop |
| C1 | Desktop Frecuente | 38.2% | Frecuentes pero superficiales |
| C2 | Weekend Warriors | 9.7% | Navegan en fin de semana |
| C3 | Mobile Users | 9.6% | Móviles, bounce alto, sesiones cortas |

## Decisiones Técnicas

- **K=4** sobre K=3: separa el segmento mobile que con K=3 queda diluido
- **KMeans** sobre Ward/GMM: silhouette comparable, mejor viabilidad de despliegue
- **LangGraph** sobre CrewAI: control explícito del grafo, fan-out nativo
- **SDK Anthropic** vía LangChain: integración con tool_use y structured output
- **MLflow**: trazabilidad de cada consulta, agentes usados, tools invocados
- **Reglas codificadas** para marketing: evita hallucination del LLM
