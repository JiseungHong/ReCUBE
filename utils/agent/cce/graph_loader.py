#!/usr/bin/env python3
"""
Graph Loader (Filtered) - Loads graphs with target file's outgoing edges removed.

Key principle: During reconstruction, the graph should not reveal:
- What the target imports (outgoing 'imports' edges)
- What the target invokes (outgoing 'invokes' edges)
- What the target inherits (outgoing 'inherits' edges)

The graph DOES reveal:
- Who calls the target (incoming edges)
- What other files import/invoke (their patterns)
- The overall graph structure
"""

import pickle
import networkx as nx
from pathlib import Path
from typing import Union, Optional
import os


def load_graph_for_reconstruction(repo_id: Union[int, str], target_file: Optional[str] = None) -> nx.MultiDiGraph:
    """
    Load dependency graph with target file's outgoing edges removed.

    This ensures the agent discovers what to import/call through exploration
    rather than being directly provided with the information.

    Args:
        repo_id: Repository ID (0-54)
        target_file: Target file being reconstructed (e.g., 'app/agent/manus.py')
                    If provided, removes outgoing edges from this file

    Returns:
        NetworkX MultiDiGraph with filtered edges
    """
    # Load the full graph
    base_dir = Path(__file__).parent.parent.parent
    graph_path = base_dir / f"data/Code_GitHub/graphs/{repo_id}.pkl"

    if not graph_path.exists():
        # Try workspace path (when running in Docker)
        graph_path = Path(f"/workspace/tools/graphs/{repo_id}.pkl")
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    # If no target file specified, return full graph
    if not target_file:
        return graph

    # Create a copy to avoid modifying original
    filtered_graph = graph.copy()

    # Find all nodes belonging to the target file
    target_nodes = []
    for node_id in filtered_graph.nodes():
        # Node can be:
        # - The file itself: 'app/agent/manus.py'
        # - A class in the file: 'app/agent/manus.py:ManusAgent'
        # - A method in the file: 'app/agent/manus.py:ManusAgent.run'
        # - A function in the file: 'app/agent/manus.py:helper_func'
        if node_id == target_file or node_id.startswith(target_file + ':'):
            target_nodes.append(node_id)

    # Remove outgoing edges from target nodes that reveal the answer
    edges_to_remove = []
    for node in target_nodes:
        for source, dest, key, edge_data in filtered_graph.edges(node, data=True, keys=True):
            edge_type = edge_data.get('type')

            # Remove edges that reveal what the target uses
            if edge_type in ['imports', 'invokes', 'inherits']:
                edges_to_remove.append((source, dest, key))

    # Remove the edges
    for source, dest, key in edges_to_remove:
        filtered_graph.remove_edge(source, dest, key)

    return filtered_graph


def save_filtered_graph(repo_id: Union[int, str], target_file: str, output_path: Optional[str] = None):
    """
    Save a filtered graph to disk for a specific reconstruction task.

    Args:
        repo_id: Repository ID
        target_file: Target file being reconstructed
        output_path: Where to save the filtered graph (default: workspace)
    """
    filtered_graph = load_graph_for_reconstruction(repo_id, target_file)

    if output_path is None:
        output_path = f"/workspace/tools/graphs/{repo_id}_filtered.pkl"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(filtered_graph, f)

    print(f"Saved filtered graph to: {output_path}")
    print(f"Removed outgoing edges from {target_file} for reconstruction task")

    # Report what was removed
    original_graph = load_graph_for_reconstruction(repo_id, None)

    removed_count = {
        'imports': 0,
        'invokes': 0,
        'inherits': 0
    }

    for node in original_graph.nodes():
        if node == target_file or node.startswith(target_file + ':'):
            for _, _, edge_data in original_graph.out_edges(node, data=True):
                edge_type = edge_data.get('type')
                if edge_type in removed_count:
                    removed_count[edge_type] += 1

    print(f"Removed edges: {removed_count}")


def get_target_from_env() -> Optional[str]:
    """
    Get the target file from environment variable.

    This is set by the task framework when starting reconstruction.
    """
    return os.environ.get('TARGET_FILE')


# Backward compatibility wrapper
def load_graph(repo_id: Union[int, str]) -> nx.MultiDiGraph:
    """
    Load graph with automatic filtering based on environment.

    If TARGET_FILE env var is set, filters out target's outgoing edges.
    Otherwise returns full graph (for analysis/debugging).
    """
    target_file = get_target_from_env()
    return load_graph_for_reconstruction(repo_id, target_file)