#!/usr/bin/env python3
"""
Graph Tools - Query dependency graphs to guide systematic exploration.

Core principles:
1. Show who USES the target (callers) - reveals usage patterns
2. Don't show what the target USES - agent discovers through exploration
3. Help agent explore systematically and efficiently
4. Prevent collection failures through validation
"""

import argparse
import ast
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Optional
from collections import defaultdict

# Import similar files module (will be available in container)
try:
    from similar_files import compute_similar_files
except ImportError:
    # Fallback if not in container environment
    compute_similar_files = None

# Lazy imports - only load when needed to avoid dependency issues
# validate_code doesn't need graph functionality
_graph_loader = None
_nx = None

def _ensure_graph_imports():
    """Lazy load graph dependencies only when needed."""
    global _graph_loader, _nx
    if _graph_loader is None:
        from graph_loader import load_graph
        import networkx as nx
        _graph_loader = sys.modules['graph_loader']
        _nx = nx
    return _graph_loader, _nx


# Simple environment setup helper
def set_target_env(target_file: str):
    """Set TARGET_FILE environment variable for graph filtering."""
    os.environ['TARGET_FILE'] = target_file


def get_repo_id():
    """
    Auto-detect repository ID from environment or context.

    Priority:
    1. REPO_ID environment variable (set by task framework)
    2. Detect from available graph files
    3. Default to '0' as fallback
    """
    # Try environment variable first
    repo_id = os.environ.get('REPO_ID')
    if repo_id:
        return repo_id

    # Try to detect from graph files
    graph_dirs = [
        Path('/workspace/tools/graphs'),
        Path(__file__).parent.parent.parent / 'data/Code_GitHub/graphs'
    ]

    for graph_dir in graph_dirs:
        if graph_dir.exists():
            pkl_files = list(graph_dir.glob('*.pkl'))
            if pkl_files:
                # Use first graph file found (usually only one)
                return pkl_files[0].stem

    # Default fallback
    print("Warning: Could not auto-detect repo_id, using default '0'")
    return '0'


def cmd_show_implementation_context(args):
    """
    Show comprehensive implementation context leveraging full graph.

    Combines multiple graph relationships to provide rich repository understanding:

    1. CALLER PATTERNS (Incoming invoke/import edges)
       - Who calls/imports this target file (EVIDENCE: r=+0.341, p=0.035*)

    2. INHERITANCE (Inherit edges)
       - Parent classes target inherits from
       - Sibling classes (other classes inheriting from same parent)

    3. MODULE CONTEXT (Directory structure)
       - Files in same directory/module
       - Common imports across siblings

    4. SIMILAR FILES (Pattern matching)
       - Files with comparable structure and patterns
    """
    # Ensure graph dependencies are loaded
    graph_loader, nx = _ensure_graph_imports()

    set_target_env(args.target_file)
    repo_id = get_repo_id()
    graph = graph_loader.load_graph(repo_id)
    target = args.target_file

    print("="*80)
    print(f"IMPLEMENTATION CONTEXT FOR: {target}")
    print("="*80)
    print()

    # ========================================================================
    # 1. CALLER PATTERNS
    # ========================================================================
    print("[1/4] CALLER PATTERNS - How your code is used")
    print("-"*80)

    callers = set()
    importers = set()
    call_details = defaultdict(lambda: defaultdict(int))  # {file: {('invokes', 'Entity'): count}}

    target_nodes = [n for n in graph.nodes()
                   if n == target or n.startswith(target + ':')]

    for node in target_nodes:
        for source, _, edge_data in graph.in_edges(node, data=True):
            edge_type = edge_data.get('type')
            source_file = source.split(':')[0]

            if source_file != target:
                if edge_type == 'invokes':
                    callers.add(source_file)
                    if ':' in node:
                        func_name = node.split(':')[1]
                        call_details[source_file][('invokes', func_name)] += 1
                elif edge_type == 'imports':
                    importers.add(source_file)
                    if ':' in node:
                        entity_name = node.split(':')[1]
                        call_details[source_file][('imports', entity_name)] += 1

    if callers or importers:
        print(f"Found {len(callers | importers)} file(s) that use your code:")
        for caller in sorted(callers | importers)[:10]:
            print(f"  {caller}")
            if caller in call_details:
                # Group by action type and deduplicate
                for (action, entity), count in sorted(call_details[caller].items())[:5]:
                    print(f"     {action}: {entity} ({count}×)")
        if len(callers | importers) > 10:
            print(f"  ... and {len(callers | importers) - 10} more files")
    else:
        print("No direct callers found (file might be entry point or unused)")
    print()

    # ========================================================================
    # 2. INHERITANCE
    # ========================================================================
    print("[2/4] INHERITANCE - Class hierarchy and siblings")
    print("-"*80)

    # Get all classes defined in target file
    target_classes = [n for n in graph.nodes()
                     if n.startswith(target + ':')
                     and graph.nodes[n].get('type') == 'class']

    # Initialize for later use
    parents = []
    siblings = set()

    if target_classes:
        for target_class in target_classes[:10]:
            class_name = target_class.split(':')[1]
            print(f"Class: {class_name}")

            # Find parent classes (outgoing inherit edges)
            parents = []
            for _, parent, edge_data in graph.out_edges(target_class, data=True):
                if edge_data.get('type') == 'inherits':
                    parents.append(parent)

            if parents:
                print(f"  Inherits from:")
                for parent in parents:
                    print(f"    {parent}")

                # Find sibling classes (other classes inheriting from same parent)
                siblings = set()
                for parent in parents:
                    for source, _, edge_data in graph.in_edges(parent, data=True):
                        if (edge_data.get('type') == 'inherits'
                            and source != target_class
                            and source.split(':')[0] != target):
                            siblings.add(source)

                if siblings:
                    print(f"  Sibling implementations ({len(siblings)} other classes):")
                    for sibling in sorted(siblings)[:10]:
                        print(f"    {sibling}")
                    if len(siblings) > 10:
                        print(f"    ... and {len(siblings) - 10} more")
            else:
                print("  No parent classes (base class or standalone)")
            print()
    else:
        print("No classes defined in target file (function-based module)")
    print()

    # ========================================================================
    # 3. MODULE CONTEXT
    # ========================================================================
    print("[3/4] MODULE CONTEXT - Directory and sibling files")
    print("-"*80)

    target_dir = os.path.dirname(target)

    # Find sibling files in same directory
    sibling_files = []
    for node in graph.nodes():
        if (graph.nodes[node].get('type') == 'file'
            and os.path.dirname(node) == target_dir
            and node != target):
            sibling_files.append(node)

    if sibling_files:
        print(f"Directory: {target_dir}/ ({len(sibling_files)} other files)")
        for sibling in sorted(sibling_files)[:10]:
            print(f"  {os.path.basename(sibling)}")
        if len(sibling_files) > 10:
            print(f"  ... and {len(sibling_files) - 10} more")
        print()

        # Analyze common imports across siblings
        common_imports = defaultdict(int)
        for sibling in sibling_files[:10]:
            sibling_nodes = [n for n in graph.nodes()
                           if n == sibling or n.startswith(sibling + ':')]
            for node in sibling_nodes:
                for _, imported, edge_data in graph.out_edges(node, data=True):
                    if edge_data.get('type') == 'imports':
                        imported_file = imported.split(':')[0]
                        if imported_file != target:
                            common_imports[imported_file] += 1

        if common_imports:
            # Show most common imports
            top_imports = sorted(common_imports.items(),
                               key=lambda x: x[1], reverse=True)[:15]
            print("Common imports in sibling files:")
            for imp, count in top_imports:
                print(f"  {imp} (used by {count}/{min(len(sibling_files), 10)} siblings)")
            print()
    else:
        print(f"No sibling files in {target_dir}/")
    print()

    # ========================================================================
    # 4. SIMILAR FILES
    # ========================================================================
    print("[4/4] SIMILAR FILES - Comparable structure and patterns")
    print("-"*80)

    if compute_similar_files is not None:
        # Use three-signal similarity scoring
        try:
            similar_results = compute_similar_files(target, graph, top_k=5, threshold=0.33)

            if similar_results:
                print(f"Found {len(similar_results)} similar file(s) using multi-signal ranking:")
                print(f"  (Signals: name overlap, structural shape, identifier BM25)")
                print()

                for file_path, combined_score, signals in similar_results:
                    print(f"  {file_path}")
                    print(f"    Score: {combined_score:.3f} (name:{signals['name_overlap']:.2f} struct:{signals['structural']:.2f} id:{signals['identifier_bm25']:.2f})")

                    # Show what this similar file defines
                    similar_nodes = [n for n in graph.nodes()
                                   if n.startswith(file_path + ':')]
                    classes = [n.split(':')[1] for n in similar_nodes
                              if graph.nodes[n].get('type') == 'class']
                    funcs = [n.split(':')[1] for n in similar_nodes
                            if graph.nodes[n].get('type') == 'function']

                    if classes:
                        print(f"    Classes: {', '.join(classes[:3])}{' ...' if len(classes) > 3 else ''}")
                    if funcs:
                        print(f"    Functions: {', '.join(funcs[:3])}{' ...' if len(funcs) > 3 else ''}")
            else:
                print("No similar files found")
        except Exception as e:
            print(f"Error computing similar files: {e}")
            print("Falling back to simple name-based matching...")
            # Fallback to old method
            _show_similar_files_fallback(target, graph)
    else:
        # Fallback if similar_files module not available
        _show_similar_files_fallback(target, graph)

    print()
    print("="*80)


def _show_similar_files_fallback(target, graph):
    """Fallback method using simple name-based matching."""
    target_basename = os.path.basename(target).replace('.py', '')
    similar_files = []

    for node in graph.nodes():
        if graph.nodes[node].get('type') == 'file' and node != target:
            node_basename = os.path.basename(node).replace('.py', '')
            # Check for common prefixes/suffixes
            if (target_basename in node_basename
                or node_basename in target_basename
                or target_basename.split('_')[0] == node_basename.split('_')[0]):
                similar_files.append(node)

    if similar_files:
        print(f"Found {len(similar_files)} file(s) with similar names:")
        for similar in sorted(similar_files)[:10]:
            print(f"  {similar}")

            # Show what this similar file defines
            similar_nodes = [n for n in graph.nodes()
                           if n.startswith(similar + ':')]
            classes = [n.split(':')[1] for n in similar_nodes
                      if graph.nodes[n].get('type') == 'class']
            if classes:
                print(f"    Defines: {', '.join(classes[:5])}")

        if len(similar_files) > 10:
            print(f"  ... and {len(similar_files) - 10} more")
    else:
        print("No files with similar names found")


def cmd_validate_code(args):
    """
    Command 4: Validate code before submission (PREVENT COLLECTION FAILURES).

    Critical for avoiding syntax errors that cause 0% test pass.
    """
    file_path = Path(args.file_path)

    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)

    with open(file_path, 'r') as f:
        code = f.read()

    print("="*80)
    print(f"VALIDATION REPORT: {file_path}")
    print("="*80)
    print()

    errors = []

    # 1. Syntax check
    print("[SYNTAX CHECK]:")
    try:
        ast.parse(code)
        print("  PASS - Python syntax is valid")
    except SyntaxError as e:
        print(f"  FAIL - SYNTAX ERROR at line {e.lineno}: {e.msg}")
        if e.text:
            print(f"       {e.text.strip()}")
        errors.append(f"Syntax error at line {e.lineno}")
    print()

    # 2. Implementation completeness
    print("[IMPLEMENTATION CHECK]:")
    not_implemented_count = code.count('raise NotImplementedError')
    if not_implemented_count > 0:
        print(f"  FAIL - Found {not_implemented_count} unimplemented functions")
        errors.append(f"{not_implemented_count} functions not implemented")
    else:
        print("  PASS - All functions appear implemented")
    print()

    # 3. Duplicate words (sed error indicator)
    print("[DUPLICATE WORD CHECK]:")
    lines = code.split('\n')
    duplicates = []
    for i, line in enumerate(lines, 1):
        words = line.strip().split()
        for j in range(len(words) - 1):
            if len(words[j]) > 2 and words[j] == words[j+1]:
                if words[j] not in ['is', 'and', 'or', 'the', 'in', 'to']:
                    duplicates.append((i, words[j]))

    if duplicates:
        print(f"  FAIL - Found duplicate words (often from sed errors):")
        for line_no, word in duplicates[:5]:
            print(f"       Line {line_no}: '{word} {word}'")
        errors.append(f"{len(duplicates)} duplicate word issues")
    else:
        print("  PASS - No duplicate patterns found")
    print()

    # Final verdict
    print("="*80)
    print("[VALIDATION RESULT]:")
    print()

    if errors:
        print("FAILED - DO NOT SUBMIT")
        print()
        print("Errors:")
        for error in errors:
            print(f"  {error}")
        sys.exit(1)
    else:
        print("PASSED - Safe to submit")




def main():
    parser = argparse.ArgumentParser(
        description='Graph tools for systematic codebase exploration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Tool capabilities (Caller-First Principle):
- show_implementation_context: Comprehensive repository understanding combining 4 perspectives
  (caller patterns, inheritance, module context, similar files)
- validate_code: Pre-submission validation to prevent failures

Workflow (EVIDENCE-BASED):
1. show_implementation_context - Get comprehensive context (RECOMMENDED FIRST)
2. validate_code - Check before submission (MANDATORY)

Examples:
  python graph_tools.py show_implementation_context --target app/main.py
  python graph_tools.py validate_code --file /workspace/app/main.py
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Command 1: show_implementation_context
    parser_context = subparsers.add_parser('show_implementation_context',
                                           help='Get comprehensive implementation context (RECOMMENDED FIRST)')
    parser_context.add_argument('--target', dest='target_file', required=True,
                               help='Target file being implemented')

    # Command 2: validate_code
    parser_validate = subparsers.add_parser('validate_code',
                                           help='Validate code before submission (MANDATORY)')
    parser_validate.add_argument('--file', dest='file_path', required=True,
                                help='File to validate')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'show_implementation_context':
            cmd_show_implementation_context(args)
        elif args.command == 'validate_code':
            cmd_validate_code(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()