"""
API REST para el sistema multi-agente de segmentación.

Endpoints:
  GET  /health           — Estado del sistema
  GET  /clusters         — Overview de los 4 clusters
  GET  /clusters/{id}    — Perfil detallado de un cluster
  POST /agent/query      — Consulta al sistema multi-agente
  GET  /traces           — Últimas ejecuciones de MLflow
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Asegurar que src/ es importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.tools import (
    get_cluster_overview, get_cluster_profile, compare_clusters,
    suggest_marketing_strategy, _clustered
)
from src.agents.graph import query as agent_query

app = FastAPI(
    title="Segmentación de Clientes — Multi-Agent API",
    version="1.0.0",
    description="Sistema multi-agente con LangGraph para análisis de segmentación de clientes."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ──────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    agents_used: list[str]
    tools_called: list[dict]
    plan: list[dict]
    total_latency_ms: int


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Estado del sistema."""
    try:
        df = _clustered()
        n_clusters = df["cluster"].nunique()
        n_visitors = len(df)
        data_ok = True
    except Exception:
        n_clusters = 0
        n_visitors = 0
        data_ok = False

    api_key_set = bool(os.getenv("ANTHROPIC_API_KEY"))
    return {
        "status": "healthy" if data_ok and api_key_set else "degraded",
        "data_loaded": data_ok,
        "n_clusters": n_clusters,
        "n_visitors": n_visitors,
        "api_key_configured": api_key_set,
        "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "version": "1.0.0",
    }


@app.get("/clusters")
async def clusters_overview():
    """Resumen de todos los clusters."""
    result = get_cluster_overview.invoke({})
    return json.loads(result)


@app.get("/clusters/{cluster_id}")
async def cluster_detail(cluster_id: int):
    """Perfil detallado de un cluster."""
    result = get_cluster_profile.invoke({"cluster_id": cluster_id})
    return json.loads(result)


@app.get("/clusters/{cluster_id}/strategy")
async def cluster_strategy(cluster_id: int):
    """Estrategia de marketing para un cluster."""
    result = suggest_marketing_strategy.invoke({"cluster_id": cluster_id})
    return json.loads(result)


@app.post("/agent/query", response_model=QueryResponse)
async def query_agent(req: QueryRequest):
    """Consulta al sistema multi-agente."""
    result = agent_query(req.query)
    return QueryResponse(**result)


@app.get("/traces")
async def get_traces():
    """Últimas ejecuciones registradas en MLflow."""
    import mlflow
    try:
        runs = mlflow.search_runs(
            experiment_names=["segmentacion_agentes"],
            max_results=20,
            order_by=["start_time DESC"]
        )
        traces = []
        for _, r in runs.iterrows():
            traces.append({
                "run_id": r.get("run_id", "")[:8],
                "query": r.get("params.query", ""),
                "agents_used": r.get("params.agents_used", "[]"),
                "in_scope": r.get("params.in_scope", "True"),
                "total_latency_ms": r.get("metrics.total_latency_ms", 0),
                "n_agents": r.get("metrics.n_agents_invoked", 0),
                "n_tools": r.get("metrics.n_tool_calls", 0),
                "start_time": str(r.get("start_time", "")),
            })
        return traces
    except Exception as e:
        return [{"error": str(e)}]


# ── Frontend estático ──────────────────────────────────────────────

FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
