"""RAG tools for code retrieval and analysis."""

from __future__ import annotations

import os
import importlib
from typing import Any

from langchain.tools import tool
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from neo4j import GraphDatabase

from langchain_core.language_models import BaseLanguageModel

from src.config import get_env
from src.llm import get_chat_model


_driver = None
_graph = None
_chain = None
_embedder = None
_cos_sim = None


def get_driver():
    """Get or create Neo4j database driver."""
    global _driver
    if _driver is not None:
        return _driver
    uri = get_env("NEO4J_URI", required=True)
    username = get_env("NEO4J_USERNAME", required=True)
    password = get_env("NEO4J_PASSWORD", required=True)
    assert uri is not None
    assert username is not None
    assert password is not None
    _driver = GraphDatabase.driver(uri, auth=(username, password))
    return _driver


def get_graph() -> Neo4jGraph:
    """Get or create Neo4j graph instance."""
    global _graph
    if _graph is not None:
        return _graph
    uri = get_env("NEO4J_URI", required=True)
    username = get_env("NEO4J_USERNAME", required=True)
    password = get_env("NEO4J_PASSWORD", required=True)
    assert uri is not None
    assert username is not None
    assert password is not None
    _graph = Neo4jGraph(url=uri, username=username, password=password)
    return _graph


def get_chain() -> GraphCypherQAChain:
    """Get or create Cypher QA chain."""
    global _chain
    if _chain is not None:
        return _chain
    graph = get_graph()
    llm = get_chat_model()
    _chain = GraphCypherQAChain.from_llm(
        llm=llm, graph=graph, verbose=True, allow_dangerous_requests=True
    )
    return _chain


def get_embedder():
    """Get or create sentence transformer embedder."""
    global _embedder
    if _embedder is not None:
        return _embedder
    model_name = get_env("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B")
    try:
        module = importlib.import_module("sentence_transformers")
        sentence_transformer = getattr(module, "SentenceTransformer")
    except Exception as exc:
        raise RuntimeError("sentence-transformers is required for embedding tools") from exc
    _embedder = sentence_transformer(model_name)
    return _embedder


def generate_embedding(text: str | list[str]):
    """Generate embedding for text or list of texts."""
    if isinstance(text, list):
        text = "".join(text)
    embedder = get_embedder()
    return embedder.encode(text)


def compute_similarity(code_embedding, query_embedding) -> float:
    """Compute cosine similarity between embeddings."""
    global _cos_sim
    if _cos_sim is None:
        try:
            module = importlib.import_module("sentence_transformers")
            util = getattr(module, "util")
            _cos_sim = util.cos_sim
        except Exception as exc:
            raise RuntimeError("sentence-transformers is required for similarity") from exc
    score = _cos_sim(code_embedding, query_embedding)
    if hasattr(score, "item"):
        return float(score.item())
    return float(score)


@tool
def general_search(query: str) -> str:
    """
    Translate natural language queries into Cypher for Neo4j search.
    Use this when other tools cannot find the answer.
    """
    chain = get_chain()
    return chain.run(query)


@tool
def get_similar_block(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Use a natural language query to find the most relevant code blocks.
    """
    query_embedding = generate_embedding(query)
    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (b:Block)
            WHERE b.embedding IS NOT NULL
            RETURN b.id AS id, b.filename AS filename, b.begin AS begin, b.end AS end, b.embedding AS embedding
            """
        )

        similarities = []
        for record in result:
            block_embedding = record["embedding"]
            sim = compute_similarity(block_embedding, query_embedding)
            similarities.append(
                {
                    "id": record["id"],
                    "filename": record["filename"],
                    "begin": record["begin"],
                    "end": record["end"],
                    "similarity": sim,
                }
            )

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]


@tool
def get_similar_module(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Use a natural language query to find the most relevant modules.
    """
    query_embedding = generate_embedding(query)
    driver = get_driver()

    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:Module)
            WHERE m.embedding IS NOT NULL
            RETURN m.id AS id, m.filename AS filename, m.embedding AS embedding
            """
        )

        similarities = []
        for record in result:
            module_embedding = record["embedding"]
            sim = compute_similarity(module_embedding, query_embedding)
            similarities.append(
                {
                    "id": record["id"],
                    "filename": record["filename"],
                    "similarity": sim,
                }
            )

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]


@tool
def get_signal_by_name(signal_name_substring: str) -> list[dict[str, Any]]:
    """
    Find signals by substring match on the name.
    """
    driver = get_driver()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (s:Signal)
            WHERE toLower(s.name) CONTAINS toLower($substring)
            RETURN s.id AS id, s.name AS name, s.filename AS filename, s.mod_belong AS mod_belong
            ORDER BY s.name
            """,
            substring=signal_name_substring,
        )

        signals = []
        for record in result:
            signals.append(
                {
                    "id": record["id"],
                    "name": record["name"],
                    "filename": record["filename"],
                    "mod_belong": record["mod_belong"],
                }
            )

        return signals


def get_upstream_nodes_by_signal_name(signal_name: str, k_layers: int = 1) -> dict[str, Any]:
    """Get upstream nodes for a signal by traversing the graph."""
    driver = get_driver()
    with driver.session() as session:
        signal_result = session.run(
            """
            MATCH (s:Signal {name: $signal_name})
            RETURN s.id AS signal_id
            """,
            signal_name=signal_name,
        )
        signal_record = signal_result.single()
        if not signal_record:
            return {}
        signal_id = signal_record["signal_id"]

    upstream_result: dict[str, Any] = {}
    with driver.session() as session:
        current_layer_nodes = [signal_id]
        for layer in range(1, k_layers + 1):
            if not current_layer_nodes:
                break
            result = session.run(
                """
                MATCH (upstream)-[r]->(current)
                WHERE current.id IN $current_ids
                RETURN upstream.id AS upstream_id,
                       upstream.name AS upstream_name,
                       labels(upstream)[0] AS upstream_type,
                       current.id AS current_id,
                       type(r) AS relationship_type
                """,
                current_ids=current_layer_nodes,
            )
            layer_nodes = []
            layer_details = []
            for record in result:
                layer_details.append(
                    {
                        "id": record["upstream_id"],
                        "name": record["upstream_name"],
                        "type": record["upstream_type"],
                        "drives_to": record["current_id"],
                        "relationship": record["relationship_type"],
                    }
                )
                layer_nodes.append(record["upstream_id"])
            upstream_result[f"layer_{layer}"] = layer_details
            current_layer_nodes = layer_nodes

    return upstream_result


def get_downstream_nodes_by_signal_name(signal_name: str, k_layers: int = 1) -> dict[str, Any]:
    """Get downstream nodes for a signal by traversing the graph."""
    driver = get_driver()
    with driver.session() as session:
        signal_result = session.run(
            """
            MATCH (s:Signal {name: $signal_name})
            RETURN s.id AS signal_id
            """,
            signal_name=signal_name,
        )
        signal_record = signal_result.single()
        if not signal_record:
            return {}
        signal_id = signal_record["signal_id"]

    downstream_result: dict[str, Any] = {}
    with driver.session() as session:
        current_layer_nodes = [signal_id]
        for layer in range(1, k_layers + 1):
            if not current_layer_nodes:
                break
            result = session.run(
                """
                MATCH (current)-[r]->(downstream)
                WHERE current.id IN $current_ids
                RETURN current.id AS current_id,
                       downstream.id AS downstream_id,
                       downstream.name AS downstream_name,
                       labels(downstream)[0] AS downstream_type,
                       type(r) AS relationship_type
                """,
                current_ids=current_layer_nodes,
            )
            layer_nodes = []
            layer_details = []
            for record in result:
                layer_details.append(
                    {
                        "id": record["downstream_id"],
                        "name": record["downstream_name"],
                        "type": record["downstream_type"],
                        "driven_by": record["current_id"],
                        "relationship": record["relationship_type"],
                    }
                )
                layer_nodes.append(record["downstream_id"])
            downstream_result[f"layer_{layer}"] = layer_details
            current_layer_nodes = layer_nodes

    return downstream_result


@tool
def get_upstream_analysis_string(signal_name: str, k_layers: int = 2) -> str:
    """
    Return a string summary of upstream nodes for a signal.
    """
    upstream_result = get_upstream_nodes_by_signal_name(signal_name, k_layers)
    result_lines = [f"Upstream Analysis for Signal: {signal_name}", "=" * 50]

    if not upstream_result:
        result_lines.append("Signal not found or no upstream nodes found.")
        return "\n".join(result_lines)

    for layer_name, nodes in upstream_result.items():
        layer_num = layer_name.split("_")[1]
        result_lines.append(f"Layer {layer_num} Upstream Nodes:")
        if nodes:
            for node in nodes:
                result_lines.append(
                    "  ← {type} '{name}' (ID: {id}) [Relationship: {relationship}]".format(
                        **node
                    )
                )
        else:
            result_lines.append("  No upstream nodes found.")
        result_lines.append("")

    return "\n".join(result_lines)


@tool
def get_downstream_analysis_string(signal_name: str, k_layers: int = 2) -> str:
    """
    Return a string summary of downstream nodes for a signal.
    """
    downstream_result = get_downstream_nodes_by_signal_name(signal_name, k_layers)
    result_lines = [f"Downstream Analysis for Signal: {signal_name}", "=" * 50]

    if not downstream_result:
        result_lines.append("Signal not found or no downstream nodes found.")
        return "\n".join(result_lines)

    for layer_name, nodes in downstream_result.items():
        layer_num = layer_name.split("_")[1]
        result_lines.append(f"Layer {layer_num} Downstream Nodes:")
        if nodes:
            for node in nodes:
                result_lines.append(
                    "  → {type} '{name}' (ID: {id}) [Relationship: {relationship}]".format(
                        **node
                    )
                )
        else:
            result_lines.append("  No downstream nodes found.")
        result_lines.append("")

    return "\n".join(result_lines)
