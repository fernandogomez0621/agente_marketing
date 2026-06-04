"""
Grafo multi-agente con LangGraph.

Diseño: Supervisor → fan-out a N agentes → Synthesis.
El supervisor NO es un simple router — descompone la consulta y puede
enviar sub-tareas a VARIOS agentes en paralelo.

Nodos:
  - supervisor:    clasifica, descompone, planifica
  - analyst:       perfiles, comparaciones, drivers (riguroso, factual)
  - statistician:  tests de significancia, validación (cauteloso)
  - strategist:    recomendaciones de marketing (orientado a negocio)
  - synthesizer:   combina outputs heterogéneos en respuesta coherente
"""
from __future__ import annotations
import os, json, time, uuid
import pandas as pd
from typing import TypedDict, Annotated, Literal, Sequence
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
import mlflow

from .tools import (ANALYST_TOOLS, STATISTICIAN_TOOLS, STRATEGIST_TOOLS,
                    ALL_TOOLS)

# ── Estado del grafo ────────────────────────────────────────────────

import operator

def _merge_lists(a: list, b: list) -> list:
    return (a or []) + (b or [])

def _merge_dicts(a: dict, b: dict) -> dict:
    return {**(a or {}), **(b or {})}

class AgentState(TypedDict):
    query: str                                              # Pregunta original
    plan: list[dict]                                        # [{agent, subtask}]
    agent_outputs: Annotated[list[dict], _merge_lists]      # Concurrent-safe
    final_answer: str                                       # Respuesta sintetizada
    trace: Annotated[dict, _merge_dicts]                    # Metadata MLflow


# ── LLM ─────────────────────────────────────────────────────────────

def _llm():
    return ChatAnthropic(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=4096,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )


# ── Prompts ─────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """Eres el orquestador de un sistema de análisis de segmentación de clientes.
Tienes 3 agentes especialistas disponibles:

1. **analyst** — Perfila clusters, compara métricas, identifica drivers. Riguroso y factual.
   Ideal para: "describe el cluster X", "compara C0 y C3", "cuáles son los clusters", "resumen general"
2. **statistician** — Valida diferencias con tests estadísticos, cuenta visitantes con filtros.
   Ideal para: "¿la diferencia es significativa?", "¿cuántos visitantes tienen bounce > 0.5?"
3. **strategist** — Genera recomendaciones de marketing basadas en reglas de negocio.
   Ideal para: "qué estrategia para C2", "cómo llegar a los mobile users"

Tu tarea: analiza la consulta del usuario y produce un plan JSON con los agentes necesarios.
PUEDES asignar VARIOS agentes si la consulta lo requiere.

Ejemplos:
- "Describe el cluster 2" → solo analyst
- "¿El cluster 0 y 2 son realmente diferentes en hits?" → analyst + statistician
- "Analiza el cluster 3 y sugiere estrategia" → analyst + strategist
- "Compara todos los clusters y recomienda acciones" → analyst + statistician + strategist
- "¿Qué es una arepa?" → fuera de scope, plan vacío

Si la consulta está FUERA del dominio de segmentación de clientes, retorna plan vacío.

Responde SOLO con JSON válido, sin markdown, sin backticks:
{"plan": [{"agent": "analyst", "subtask": "describe cluster 2"}, ...], "reasoning": "..."}
Si fuera de scope: {"plan": [], "reasoning": "fuera de dominio", "fallback": "mensaje amable"}
"""

ANALYST_PROMPT = """Eres un analista de datos riguroso especializado en segmentación de clientes.
Tu trabajo es perfilar clusters, comparar métricas y dar insights basados en datos.

Reglas:
- Cita números específicos de los datos. Nunca inventes.
- Usa z-scores para contextualizar si un valor es alto o bajo vs el global.
- Si comparas clusters, menciona effect sizes, no solo promedios.
- Sé conciso pero completo.

Los 4 clusters son:
- C0: Desktop Engaged (42.5%) - alto engagement desktop
- C1: Desktop Frecuente (38.2%) - frecuentes pero superficiales
- C2: Weekend Warriors (9.7%) - navegan en fin de semana
- C3: Mobile Users (9.6%) - usuarios móviles
"""

STATISTICIAN_PROMPT = """Eres un estadístico cauteloso. Validas afirmaciones con tests formales.

Variables numéricas disponibles para tests:
  n_sessions, totals.hits, totals.pageviews, bounce_prop, weekend_prop,
  hits_per_session, pageviews_per_session, hits_per_pageview, hour.

IMPORTANTE: "bounce" = bounce_prop, "hits" = totals.hits, "pageviews" = totals.pageviews.

Con n≈10000, todo p-value sale ~0, así que siempre reportas effect size además de p-value.
Interpretas Mann-Whitney U con rank-biserial r: >0.5 grande, >0.3 mediano, >0.1 pequeño, <0.1 trivial.
Si una diferencia es trivial en effect size aunque sea significativa, dilo claramente.
No afirmes nada sin evidencia del test."""

STRATEGIST_PROMPT = """Eres un estratega de marketing digital orientado a resultados.
Das recomendaciones accionables basadas en los perfiles de los clusters.
IMPORTANTE: tus recomendaciones se basan en las reglas de negocio pre-codificadas
que obtienes del tool suggest_marketing_strategy. No inventes canales ni tácticas
que no estén en los datos — complementa con tu conocimiento pero la base es lo que
retorna el tool.
Estructura tus recomendaciones en: canal, timing, mensaje, KPIs, presupuesto."""

SYNTHESIS_PROMPT = """Eres el sintetizador del sistema. Recibes outputs de varios agentes
especialistas y los combinas en UNA sola respuesta coherente para el usuario.

Reglas:
- No repitas información entre secciones.
- Mantén el rigor del analyst, la cautela del statistician, y la acción del strategist.
- Si hay conflictos entre agentes, menciónalos.
- Responde en español, de forma profesional pero accesible.
- Estructura la respuesta con secciones claras si hay múltiples agentes.
"""


# ── Nodos del grafo ─────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    """Analiza la query, clasifica scope, descompone en sub-tareas, asigna agentes."""
    t0 = time.time()
    llm = _llm()
    response = llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {state['query']}")
    ])
    # Parsear JSON del supervisor
    try:
        text = response.content.strip()
        # Limpiar posibles backticks
        if text.startswith("```"): text = text.split("```")[1]
        if text.startswith("json"): text = text[4:]
        result = json.loads(text)
    except json.JSONDecodeError:
        result = {"plan": [{"agent": "analyst", "subtask": state["query"]}],
                  "reasoning": "fallback a analyst"}

    plan = result.get("plan", [])
    trace = {
        "supervisor_reasoning": result.get("reasoning", ""),
        "agents_planned": [p["agent"] for p in plan],
        "is_in_scope": len(plan) > 0,
        "supervisor_latency_ms": int((time.time()-t0)*1000),
    }
    if not plan:
        fallback = result.get("fallback", "Lo siento, solo puedo responder preguntas sobre la segmentación de clientes.")
        return {"plan": [], "final_answer": fallback, "trace": trace, "agent_outputs": []}
    return {"plan": plan, "trace": trace, "agent_outputs": []}


def _run_agent(agent_name: str, prompt: str, tools: list, subtask: str, state: AgentState) -> dict:
    """Ejecuta un agente con su prompt, tools y subtask. Retorna el output."""
    t0 = time.time()
    llm = _llm()
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=subtask)
    ]
    tools_called = []

    # Loop de tool_use (max 5 iteraciones para evitar loops infinitos)
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        if not response.tool_calls:
            break
        messages.append(response)
        for tc in response.tool_calls:
            tool_fn = {t.name: t for t in tools}.get(tc["name"])
            if tool_fn:
                try:
                    result = tool_fn.invoke(tc["args"])
                except Exception as e:
                    result = json.dumps({"error": f"Tool {tc['name']} falló: {str(e)}"})
                tools_called.append({"name": tc["name"], "args": tc["args"]})
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Extraer texto de la respuesta (puede ser string o lista de blocks)
    content = response.content if hasattr(response, 'content') else str(response)
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
            if not (isinstance(block, dict) and block.get("type") == "tool_use")
        ) or "Sin respuesta textual."

    return {
        "agent": agent_name,
        "subtask": subtask,
        "response": content,
        "tools_called": tools_called,
        "latency_ms": int((time.time()-t0)*1000),
    }


def analyst_node(state: AgentState) -> dict:
    """Ejecuta el agente analyst para cada subtask asignada."""
    outputs = []
    for task in state["plan"]:
        if task["agent"] == "analyst":
            out = _run_agent("analyst", ANALYST_PROMPT, ANALYST_TOOLS,
                           task["subtask"], state)
            outputs.append(out)
    return {"agent_outputs": outputs}


def statistician_node(state: AgentState) -> dict:
    """Ejecuta el agente statistician."""
    outputs = []
    for task in state["plan"]:
        if task["agent"] == "statistician":
            out = _run_agent("statistician", STATISTICIAN_PROMPT, STATISTICIAN_TOOLS,
                           task["subtask"], state)
            outputs.append(out)
    return {"agent_outputs": outputs}


def strategist_node(state: AgentState) -> dict:
    """Ejecuta el agente strategist."""
    outputs = []
    for task in state["plan"]:
        if task["agent"] == "strategist":
            out = _run_agent("strategist", STRATEGIST_PROMPT, STRATEGIST_TOOLS,
                           task["subtask"], state)
            outputs.append(out)
    return {"agent_outputs": outputs}


def synthesizer_node(state: AgentState) -> dict:
    """Combina outputs de todos los agentes en una respuesta coherente."""
    outputs = state.get("agent_outputs", [])
    if not outputs:
        return {"final_answer": state.get("final_answer", "Sin resultados.")}
    if len(outputs) == 1:
        return {"final_answer": outputs[0]["response"]}

    # Múltiples agentes → sintetizar
    t0 = time.time()
    llm = _llm()
    agent_summaries = "\n\n".join([
        f"=== {o['agent'].upper()} (subtask: {o['subtask']}) ===\n{o['response']}"
        for o in outputs
    ])
    response = llm.invoke([
        SystemMessage(content=SYNTHESIS_PROMPT),
        HumanMessage(content=f"Query original: {state['query']}\n\n"
                             f"Outputs de los agentes:\n{agent_summaries}")
    ])
    content = response.content
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ) or "Sin respuesta."
    trace = dict(state.get("trace", {}))
    trace["synthesis_latency_ms"] = int((time.time()-t0)*1000)
    return {"final_answer": content, "trace": trace}


# ── Routing ─────────────────────────────────────────────────────────

def route_after_supervisor(state: AgentState) -> list[str]:
    """Determina a qué nodos ir basándose en el plan del supervisor."""
    if not state.get("plan"):
        return ["synthesizer"]

    agents_needed = set(task["agent"] for task in state["plan"])
    route = []
    if "analyst" in agents_needed:
        route.append("analyst")
    if "statistician" in agents_needed:
        route.append("statistician")
    if "strategist" in agents_needed:
        route.append("strategist")
    return route if route else ["synthesizer"]


# ── Construcción del grafo ──────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construye y compila el grafo multi-agente."""
    graph = StateGraph(AgentState)

    # Nodos
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("statistician", statistician_node)
    graph.add_node("strategist", strategist_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Entry point
    graph.set_entry_point("supervisor")

    # Conditional fan-out desde supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "analyst": "analyst",
            "statistician": "statistician",
            "strategist": "strategist",
            "synthesizer": "synthesizer",
        }
    )

    # Todos los agentes → synthesizer
    graph.add_edge("analyst", "synthesizer")
    graph.add_edge("statistician", "synthesizer")
    graph.add_edge("strategist", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph.compile()


# ── Invocación con MLflow tracing ───────────────────────────────────

# Activar autolog de LangChain → captura tool calls, mensajes, traces
mlflow.langchain.autolog(log_traces=True)

# Flag para registrar el modelo solo una vez
_model_registered = False

def _register_model_once():
    """Registra el agente como modelo en MLflow Registry."""
    global _model_registered
    if _model_registered:
        return
    try:
        mlflow.set_experiment("segmentacion_agentes")

        # Wrapper pyfunc para el agente
        class SegmentationAgent(mlflow.pyfunc.PythonModel):
            def predict(self, context, model_input):
                """Ejecuta una query contra el sistema multi-agente."""
                if isinstance(model_input, pd.DataFrame):
                    q = model_input.iloc[0]["query"]
                else:
                    q = str(model_input)
                app = build_graph()
                result = app.invoke({
                    "query": q, "plan": [], "agent_outputs": [],
                    "final_answer": "", "trace": {},
                })
                return result.get("final_answer", "")

        with mlflow.start_run(run_name="register_agent_v1"):
            model_info = mlflow.pyfunc.log_model(
                artifact_path="segmentation_agent",
                python_model=SegmentationAgent(),
                registered_model_name="segmentation-multi-agent",
                input_example=pd.DataFrame([{"query": "¿Cuántos clusters hay?"}]),
            )
            mlflow.set_tag("model_type", "multi-agent-langgraph")
            mlflow.set_tag("agents", "analyst, statistician, strategist")
            mlflow.set_tag("n_tools", "9")
            mlflow.set_tag("framework", "LangGraph + LangChain + Anthropic")
            mlflow.log_param("supervisor_model", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"))
            mlflow.log_param("n_agents", 3)
            mlflow.log_param("n_tools", 9)
        _model_registered = True
    except Exception as e:
        print(f"Warning: model registration failed: {e}")
        _model_registered = True

def query(user_query: str, experiment_name: str = "segmentacion_agentes") -> dict:
    """Punto de entrada principal. Ejecuta el grafo y registra en MLflow."""
    _register_model_once()
    mlflow.set_experiment(experiment_name)

    app = build_graph()

    with mlflow.start_run(run_name=f"query_{uuid.uuid4().hex[:8]}"):
        mlflow.set_tag("query", user_query[:200])
        t0 = time.time()

        result = app.invoke({
            "query": user_query,
            "plan": [],
            "agent_outputs": [],
            "final_answer": "",
            "trace": {},
        })

        total_ms = int((time.time()-t0)*1000)
        trace = result.get("trace", {})

        # Métricas custom (complementan el autolog)
        mlflow.log_param("query_text", user_query[:250])
        mlflow.log_param("agents_planned", str(trace.get("agents_planned", [])))
        mlflow.log_metric("total_latency_ms", total_ms)
        mlflow.log_metric("supervisor_latency_ms", trace.get("supervisor_latency_ms", 0))
        mlflow.log_metric("n_agents_invoked", len(result.get("agent_outputs", [])))
        mlflow.log_metric("n_tool_calls", sum(
            len(o.get("tools_called", [])) for o in result.get("agent_outputs", [])
        ))

        for i, o in enumerate(result.get("agent_outputs", [])):
            mlflow.log_metric(f"agent_{i}_latency_ms", o.get("latency_ms", 0))
            mlflow.set_tag(f"agent_{i}_name", o.get("agent", ""))

    return {
        "answer": result.get("final_answer", ""),
        "plan": result.get("plan", []),
        "agents_used": [o["agent"] for o in result.get("agent_outputs", [])],
        "tools_called": [t for o in result.get("agent_outputs", []) for t in o.get("tools_called", [])],
        "total_latency_ms": total_ms,
    }
