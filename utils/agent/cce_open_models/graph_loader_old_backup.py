#!/usr/bin/env python3
"""
Graph Loader - Utilities for loading and querying dependency graphs.

Provides functions to:
- Load pre-built graphs
- Query file dependencies
- Search for entities by name
- Export tree structure
"""

import pickle
import networkx as nx
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Union
import os

# Import filtered loader for reconstruction tasks
from .graph_loader_filtered import load_graph_for_reconstruction


def load_graph(repo_id: Union[int, str]) -> nx.MultiDiGraph:
    """Load pre-built dependency graph.

    IMPORTANT: Automatically filters out target's outgoing edges if TARGET_FILE
    environment variable is set (for reconstruction tasks).

    Args:
        repo_id: Repository ID (0-54)

    Returns:
        NetworkX MultiDiGraph (filtered if reconstructing a target file)

    Raises:
        FileNotFoundError: If graph file doesn't exist
    """
    # Check if we're in reconstruction mode
    target_file = os.environ.get('TARGET_FILE')

    # Use filtered loader for reconstruction
    return load_graph_for_reconstruction(repo_id, target_file)


def get_file_dependencies(
    graph: nx.MultiDiGraph,
    file_path: str,
    depth: int = 2,
    edge_types: List[str] = None
) -> Dict[str, List[str]]:
    """Get dependencies for a target file.

    Args:
        graph: Dependency graph
        file_path: Target file (e.g., 'app/agent/manus.py')
        depth: How many hops to traverse (1-3)
        edge_types: Which edge types to follow. Default: ['imports', 'invokes']

    Returns:
        Dictionary with categorized dependencies:
        {
            'imports': ['app/config.py', ...],
            'invokes': ['app/logger.py:setup_logger', ...],
            'inherits': ['app/agent/base.py:BaseAgent', ...],
            'all_related_files': ['app/config.py', 'app/logger.py', ...]
        }
    """
    if edge_types is None:
        edge_types = ['imports', 'invokes']

    if file_path not in graph:
        return {
            'imports': [],
            'invokes': [],
            'inherits': [],
            'all_related_files': []
        }

    # Track dependencies by type
    dependencies = {
        'imports': set(),
        'invokes': set(),
        'inherits': set()
    }

    # BFS traversal
    visited = set()
    queue = [(file_path, 0)]  # (node_id, current_depth)

    while queue:
        current, current_depth = queue.pop(0)

        if current in visited or current_depth > depth:
            continue

        visited.add(current)

        # Get outgoing edges
        for _, target, edge_data in graph.out_edges(current, data=True):
            edge_type = edge_data.get('type')

            # Skip if not in requested edge types
            if edge_type not in edge_types:
                continue

            # Add to appropriate category
            if edge_type in dependencies:
                dependencies[edge_type].add(target)

            # Continue traversal
            if current_depth < depth:
                queue.append((target, current_depth + 1))

    # Extract unique files from all dependencies
    all_files = set()
    for deps in dependencies.values():
        for dep in deps:
            # Extract file path from node ID (format: file.py or file.py:Class.method)
            file = dep.split(':')[0]
            if file != file_path:  # Don't include the target file itself
                all_files.add(file)

    return {
        'imports': sorted(list(dependencies['imports'])),
        'invokes': sorted(list(dependencies['invokes'])),
        'inherits': sorted(list(dependencies['inherits'])),
        'all_related_files': sorted(list(all_files))
    }


def search_entities(
    graph: nx.MultiDiGraph,
    entity_names: List[str]
) -> Dict[str, List[str]]:
    """Search for classes/functions by name.

    Args:
        graph: Dependency graph
        entity_names: List of entity names to search for

    Returns:
        Dictionary mapping entity names to node IDs:
        {
            'ConfigLoader': ['app/config.py:ConfigLoader'],
            'setup_logger': ['app/logger.py:setup_logger'],
            ...
        }
    """
    results = defaultdict(list)

    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('type')

        # Only search in class and function nodes
        if node_type not in ['class', 'function']:
            continue

        # Extract entity name from node ID
        # Format: file.py:EntityName or file.py:Class.method
        if ':' in node_id:
            entity_name = node_id.split(':')[1].split('.')[-1]

            # Check if this entity matches any search term
            for search_name in entity_names:
                if entity_name == search_name:
                    results[search_name].append(node_id)

    return dict(results)


def export_tree_structure(
    graph: nx.MultiDiGraph,
    max_depth: int = 3,
    start_node: str = '/'
) -> str:
    """Export tree view of repository structure.

    Args:
        graph: Dependency graph
        max_depth: Maximum depth to display
        start_node: Root node to start from (default: '/')

    Returns:
        Multi-line string with tree structure:
        - [directory] /
          - [directory] app
            - [file] app/main.py
              - [class] app/main.py:Agent
                - [function] app/main.py:Agent.run
    """
    lines = []

    def traverse(node_id: str, depth: int, prefix: str = ""):
        """Recursive DFS traversal."""
        if depth > max_depth:
            return

        # Get node type
        node_data = graph.nodes.get(node_id, {})
        node_type = node_data.get('type', 'unknown')

        # Format node display
        if depth == 0:
            lines.append(f"- [{node_type}] {node_id}")
        else:
            lines.append(f"{prefix}- [{node_type}] {node_id}")

        # Get children (nodes connected by 'contains' edges)
        children = []
        for _, target, edge_data in graph.out_edges(node_id, data=True):
            if edge_data.get('type') == 'contains':
                children.append(target)

        # Sort children for consistent output
        children.sort()

        # Traverse children
        for i, child in enumerate(children):
            is_last = (i == len(children) - 1)
            child_prefix = prefix + ("  " if is_last else "  ")
            traverse(child, depth + 1, child_prefix)

    # Start traversal from root
    if start_node in graph:
        traverse(start_node, 0)
    else:
        lines.append(f"Error: Node '{start_node}' not found in graph")

    return '\n'.join(lines)


def get_graph_statistics(graph: nx.MultiDiGraph) -> Dict:
    """Get statistics about the graph.

    Args:
        graph: Dependency graph

    Returns:
        Dictionary with statistics
    """
    from collections import Counter

    node_types = Counter([data['type'] for _, data in graph.nodes(data=True)])
    edge_types = Counter([data['type'] for _, _, data in graph.edges(data=True)])

    return {
        'total_nodes': graph.number_of_nodes(),
        'total_edges': graph.number_of_edges(),
        'node_types': dict(node_types),
        'edge_types': dict(edge_types)
    }


def get_entity_code(graph: nx.MultiDiGraph, node_id: str) -> str:
    """Get the source code for a specific entity.

    Args:
        graph: Dependency graph
        node_id: Node ID (e.g., 'app/main.py:Agent' or 'app/main.py')

    Returns:
        Source code string, or empty string if not found
    """
    if node_id not in graph:
        return ""

    node_data = graph.nodes[node_id]
    return node_data.get('code', '')


def find_files_using_entity(graph: nx.MultiDiGraph, entity_id: str) -> List[str]:
    """Find all files that import or invoke a specific entity.

    Args:
        graph: Dependency graph
        entity_id: Entity node ID (e.g., 'app/config.py:ConfigLoader')

    Returns:
        List of file paths that use this entity
    """
    using_files = set()

    # Find all nodes that have edges to this entity
    for source, target, edge_data in graph.edges(data=True):
        if target == entity_id:
            edge_type = edge_data.get('type')
            if edge_type in ['imports', 'invokes']:
                # Extract file from source node
                file_path = source.split(':')[0]
                using_files.add(file_path)

    return sorted(list(using_files))
