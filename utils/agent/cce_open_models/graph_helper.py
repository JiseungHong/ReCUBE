#!/usr/bin/env python3
"""
Graph Helper - Simplified Python API for graph queries in code execution.

This module provides easy-to-use functions that LLMs can call directly
in Python code blocks, similar to LocAgent's IPython execution approach.

Usage in agent Python code:
    from graph_helper import load_repo_graph, search_entities, get_dependencies, ...
    graph = load_repo_graph()
    deps = get_dependencies(graph, 'app/main.py', depth=2)
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Set, Optional
import networkx as nx


# Auto-detect repo_id from environment or current directory
def _get_repo_id() -> str:
    """Auto-detect repository ID from environment."""
    # Try environment variable first
    repo_id = os.environ.get('REPO_ID')
    if repo_id:
        return repo_id

    # Try to infer from available graph files
    graph_dirs = [
        Path('/workspace/tools/graphs'),
        Path(__file__).parent.parent.parent / 'data/Code_GitHub/graphs'
    ]

    for graphs_dir in graph_dirs:
        if graphs_dir.exists():
            pkl_files = list(graphs_dir.glob('*.pkl'))
            # Filter out _filtered versions if any
            pkl_files = [f for f in pkl_files if not f.stem.endswith('_filtered')]
            if pkl_files:
                # Use the first (and likely only) graph file
                return pkl_files[0].stem

    # Default fallback
    return '0'


def load_repo_graph(repo_id: Optional[str] = None) -> nx.MultiDiGraph:
    """
    Load the dependency graph for the current repository.

    Args:
        repo_id: Repository ID (auto-detected if not provided)

    Returns:
        NetworkX MultiDiGraph with code dependencies

    Example:
        graph = load_repo_graph()
        print(f"Nodes: {graph.number_of_nodes()}")
    """
    if repo_id is None:
        repo_id = _get_repo_id()

    graph_path = Path(f'/workspace/tools/graphs/{repo_id}.pkl')

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found: {graph_path}")

    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def search_entities(graph: nx.MultiDiGraph, names: List[str]) -> Dict[str, List[str]]:
    """
    Search for classes/functions by name.

    Args:
        graph: Dependency graph
        names: List of entity names to search for

    Returns:
        Dict mapping names to list of node IDs

    Example:
        graph = load_repo_graph()
        results = search_entities(graph, ['Config', 'Logger', 'BaseAgent'])
        for name, node_ids in results.items():
            print(f"{name}: {node_ids}")
    """
    results = {}

    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get('type')

        # Only search in class and function nodes
        if node_type not in ['class', 'function']:
            continue

        # Extract entity name from node ID
        if ':' in node_id:
            entity_name = node_id.split(':')[1].split('.')[-1]

            for search_name in names:
                if entity_name == search_name:
                    if search_name not in results:
                        results[search_name] = []
                    results[search_name].append(node_id)

    return results


def get_dependencies(
    graph: nx.MultiDiGraph,
    file_path: str,
    depth: int = 2,
    edge_types: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Get dependencies for a file (imports, invokes, inherits).

    Args:
        graph: Dependency graph
        file_path: Target file path
        depth: How many hops to traverse
        edge_types: Which edges to follow (default: ['imports', 'invokes'])

    Returns:
        Dict with 'imports', 'invokes', 'inherits', 'all_related_files'

    Example:
        graph = load_repo_graph()
        deps = get_dependencies(graph, 'app/main.py', depth=2)
        print(f"Imports: {deps['imports']}")
        print(f"Related files: {deps['all_related_files']}")
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
    queue = [(file_path, 0)]

    while queue:
        current, current_depth = queue.pop(0)

        if current in visited or current_depth > depth:
            continue

        visited.add(current)

        # Get outgoing edges
        for _, target, edge_data in graph.out_edges(current, data=True):
            edge_type = edge_data.get('type')

            if edge_type not in edge_types:
                continue

            if edge_type in dependencies:
                dependencies[edge_type].add(target)

            if current_depth < depth:
                queue.append((target, current_depth + 1))

    # Extract unique files
    all_files = set()
    for deps in dependencies.values():
        for dep in deps:
            file = dep.split(':')[0]
            if file != file_path:
                all_files.add(file)

    return {
        'imports': sorted(list(dependencies['imports'])),
        'invokes': sorted(list(dependencies['invokes'])),
        'inherits': sorted(list(dependencies['inherits'])),
        'all_related_files': sorted(list(all_files))
    }


def get_file_content(graph: nx.MultiDiGraph, file_path: str) -> str:
    """
    Get the full source code of a file from the graph.

    Args:
        graph: Dependency graph
        file_path: File path (e.g., 'app/main.py')

    Returns:
        File content as string

    Example:
        graph = load_repo_graph()
        code = get_file_content(graph, 'app/config.py')
        print(code)
    """
    if file_path not in graph:
        return ""

    node_data = graph.nodes[file_path]
    return node_data.get('code', '')


def get_entity_code(graph: nx.MultiDiGraph, node_id: str) -> str:
    """
    Get source code for a specific class or function.

    Args:
        graph: Dependency graph
        node_id: Node ID (e.g., 'app/config.py:Config')

    Returns:
        Entity source code

    Example:
        graph = load_repo_graph()
        code = get_entity_code(graph, 'app/config.py:Config')
        print(code)
    """
    if node_id not in graph:
        return ""

    node_data = graph.nodes[node_id]
    return node_data.get('code', '')


def list_files(graph: nx.MultiDiGraph) -> List[str]:
    """
    List all Python files in the repository.

    Args:
        graph: Dependency graph

    Returns:
        Sorted list of file paths

    Example:
        graph = load_repo_graph()
        files = list_files(graph)
        for f in files:
            print(f)
    """
    files = []
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('type') == 'file':
            files.append(node_id)
    return sorted(files)


def list_classes(graph: nx.MultiDiGraph, file_path: Optional[str] = None) -> List[str]:
    """
    List all classes (optionally filtered by file).

    Args:
        graph: Dependency graph
        file_path: Optional file path to filter by

    Returns:
        List of class node IDs

    Example:
        graph = load_repo_graph()
        classes = list_classes(graph, 'app/config.py')
        for c in classes:
            print(c)
    """
    classes = []
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('type') == 'class':
            if file_path is None or node_id.startswith(file_path + ':'):
                classes.append(node_id)
    return sorted(classes)


def list_functions(graph: nx.MultiDiGraph, file_path: Optional[str] = None) -> List[str]:
    """
    List all functions (optionally filtered by file).

    Args:
        graph: Dependency graph
        file_path: Optional file path to filter by

    Returns:
        List of function node IDs

    Example:
        graph = load_repo_graph()
        funcs = list_functions(graph, 'app/utils.py')
        for f in funcs:
            print(f)
    """
    functions = []
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get('type') == 'function':
            if file_path is None or node_id.startswith(file_path + ':'):
                functions.append(node_id)
    return sorted(functions)


def find_usages(graph: nx.MultiDiGraph, entity_id: str) -> List[str]:
    """
    Find all files that import or invoke an entity.

    Args:
        graph: Dependency graph
        entity_id: Entity node ID

    Returns:
        List of file paths that use this entity

    Example:
        graph = load_repo_graph()
        usages = find_usages(graph, 'app/config.py:Config')
        print(f"Files using Config: {usages}")
    """
    using_files = set()

    for source, target, edge_data in graph.edges(data=True):
        if target == entity_id:
            edge_type = edge_data.get('type')
            if edge_type in ['imports', 'invokes']:
                file_path = source.split(':')[0]
                using_files.add(file_path)

    return sorted(list(using_files))


def get_imports_for_file(graph: nx.MultiDiGraph, file_path: str) -> List[Dict[str, str]]:
    """
    Get all import statements for a file with details.

    Args:
        graph: Dependency graph
        file_path: File path

    Returns:
        List of dicts with 'target' and 'alias' keys

    Example:
        graph = load_repo_graph()
        imports = get_imports_for_file(graph, 'app/main.py')
        for imp in imports:
            print(f"from {imp['target'].split(':')[0]} import {imp['target'].split(':')[1]} as {imp.get('alias', imp['target'].split(':')[1])}")
    """
    if file_path not in graph:
        return []

    imports = []
    for _, target, edge_data in graph.out_edges(file_path, data=True):
        if edge_data.get('type') == 'imports':
            imports.append({
                'target': target,
                'alias': edge_data.get('alias')
            })

    return imports


# Quick helper for common pattern: load graph and get dependencies
def quick_deps(file_path: str, depth: int = 2) -> Dict[str, List[str]]:
    """
    Quick helper: Load graph and get dependencies in one call.

    Args:
        file_path: Target file
        depth: Traversal depth

    Returns:
        Dependencies dict

    Example:
        deps = quick_deps('app/main.py')
        print(f"Need to import: {deps['imports']}")
    """
    graph = load_repo_graph()
    return get_dependencies(graph, file_path, depth)


# Quick helper for searching entities
def quick_search(*names: str) -> Dict[str, List[str]]:
    """
    Quick helper: Load graph and search entities in one call.

    Args:
        *names: Entity names to search for

    Returns:
        Search results dict

    Example:
        results = quick_search('Config', 'Logger')
        print(results)
    """
    graph = load_repo_graph()
    return search_entities(graph, list(names))
