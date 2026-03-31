from langgraph.graph import END, StateGraph

from src.agents.nodes.cluster_node import cluster_node
from src.agents.nodes.evaluate_node import evaluate_node
from src.agents.nodes.ingest_node import ingest_node
from src.agents.nodes.interpret_node import interpret_node
from src.agents.nodes.optimize_k_node import optimize_k_node
from src.agents.nodes.preprocess_node import preprocess_node
from src.agents.nodes.profile_node import profile_node
from src.agents.nodes.refinement_node import refinement_node
from src.agents.nodes.select_node import select_node
from src.agents.routing import should_refine
from src.models.state import PipelineState


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("profile", profile_node)
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("optimize_k", optimize_k_node)
    graph.add_node("select", select_node)
    graph.add_node("cluster", cluster_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("interpret", interpret_node)
    graph.add_node("refinement", refinement_node)

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "profile")
    graph.add_edge("profile", "preprocess")
    graph.add_edge("preprocess", "optimize_k")
    graph.add_edge("optimize_k", "select")
    graph.add_edge("select", "cluster")
    graph.add_edge("cluster", "evaluate")
    graph.add_edge("evaluate", "interpret")
    graph.add_edge("interpret", "refinement")

    graph.add_conditional_edges("refinement", should_refine, {
        "cluster": "cluster",
        "end": END,
    })

    return graph


def compile_graph():
    return build_graph().compile()
