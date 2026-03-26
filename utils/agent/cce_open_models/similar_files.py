#!/usr/bin/env python3
"""
Similar Files Retrieval using Three-Signal Scoring.

Implements a sophisticated similarity metric combining:
1. File name sub-token overlap (1/3 weight)
2. Structural shape similarity (1/3 weight)
3. BM25 on identifiers (1/3 weight)
"""

import ast
import os
import math
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter


def tokenize_filename(filename: str) -> Set[str]:
    """
    Tokenize filename by splitting on underscores.

    Args:
        filename: File basename (without extension)

    Returns:
        Set of lowercase tokens

    Example:
        'yaml_parser' -> {'yaml', 'parser'}
    """
    # Remove .py extension if present
    stem = filename.replace('.py', '')
    # Split on underscores and convert to lowercase
    tokens = set(token.lower() for token in stem.split('_') if token)
    return tokens


def overlap_coefficient(set1: Set[str], set2: Set[str]) -> float:
    """
    Compute overlap coefficient: |intersection| / min(|set1|, |set2|).

    Returns 1.0 if one set is subset of another, 0.0 if disjoint.
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    min_size = min(len(set1), len(set2))

    return intersection / min_size if min_size > 0 else 0.0


def extract_structural_features(code: str) -> Dict[str, int]:
    """
    Extract structural features from Python code via AST.

    Returns:
        Dictionary with keys: 'classes', 'functions', 'depth'
    """
    features = {'classes': 0, 'functions': 0, 'depth': 0}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return features

    # Count classes and functions
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            features['classes'] += 1
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            features['functions'] += 1

    # Compute max nesting depth
    def compute_depth(node, current_depth=0):
        max_depth = current_depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                child_depth = compute_depth(child, current_depth + 1)
                max_depth = max(max_depth, child_depth)
            else:
                child_depth = compute_depth(child, current_depth)
                max_depth = max(max_depth, child_depth)
        return max_depth

    features['depth'] = compute_depth(tree)

    return features


def structural_similarity(features1: Dict[str, int], features2: Dict[str, int]) -> float:
    """
    Compute structural shape similarity.

    Formula: 0.4 × classes_sim + 0.4 × functions_sim + 0.2 × depth_sim
    Where each sim = min(x, y) / max(x, y), or 1.0 if both are 0.
    """
    def ratio_similarity(x, y):
        if x == 0 and y == 0:
            return 1.0
        if x == 0 or y == 0:
            return 0.0
        return min(x, y) / max(x, y)

    class_sim = ratio_similarity(features1['classes'], features2['classes'])
    func_sim = ratio_similarity(features1['functions'], features2['functions'])
    depth_sim = ratio_similarity(features1['depth'], features2['depth'])

    return 0.4 * class_sim + 0.4 * func_sim + 0.2 * depth_sim


def extract_identifiers(code: str) -> List[str]:
    """
    Extract all identifiers from Python code: function names, class names, import names.

    Each identifier is split on _ and lowercased.

    Returns:
        List of tokens (with duplicates for frequency)
    """
    tokens = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return tokens

    for node in ast.walk(tree):
        # Function and class names
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name_tokens = node.name.split('_')
            tokens.extend(token.lower() for token in name_tokens if token)

        # Import names
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module_tokens = alias.name.split('.')
                for module_token in module_tokens:
                    sub_tokens = module_token.split('_')
                    tokens.extend(token.lower() for token in sub_tokens if token)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_tokens = node.module.split('.')
                for module_token in module_tokens:
                    sub_tokens = module_token.split('_')
                    tokens.extend(token.lower() for token in sub_tokens if token)

    return tokens


class BM25:
    """
    BM25 ranking function for document retrieval.

    Parameters k1 and b follow standard BM25 defaults.
    """

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 with a corpus of token lists.

        Args:
            corpus: List of documents, where each document is a list of tokens
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling length normalization
        """
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.corpus_size = len(corpus)

        # Compute document frequencies
        self.df = defaultdict(int)  # Document frequency
        for doc in corpus:
            unique_tokens = set(doc)
            for token in unique_tokens:
                self.df[token] += 1

        # Compute IDF
        self.idf = {}
        for token, df in self.df.items():
            # Standard IDF formula: log((N - df + 0.5) / (df + 0.5))
            self.idf[token] = math.log((self.corpus_size - df + 0.5) / (df + 0.5))

        # Compute average document length
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0

        # Precompute document lengths
        self.doc_lengths = [len(doc) for doc in corpus]

    def score(self, query: List[str], doc_idx: int) -> float:
        """
        Compute BM25 score for a query against a specific document.

        Args:
            query: List of query tokens
            doc_idx: Index of document in corpus

        Returns:
            BM25 score (higher is better)
        """
        score = 0.0
        doc = self.corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        # Count term frequencies in document
        tf = Counter(doc)

        for token in query:
            if token not in self.idf:
                continue  # Token not in corpus

            idf = self.idf[token]
            freq = tf.get(token, 0)

            # BM25 formula
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))

            score += idf * (numerator / denominator)

        return score

    def get_scores(self, query: List[str]) -> List[float]:
        """
        Get BM25 scores for query against all documents.

        Returns:
            List of scores, one per document in corpus
        """
        return [self.score(query, i) for i in range(self.corpus_size)]


def compute_similar_files(
    target_file: str,
    graph,
    top_k: int = 5,
    threshold: float = 0.33
) -> List[Tuple[str, float, Dict[str, float]]]:
    """
    Find most similar files using three-signal scoring.

    Args:
        target_file: Path to target file
        graph: NetworkX graph with file nodes containing 'code' attribute
        top_k: Number of top similar files to return (default: 5)
        threshold: Minimum combined score to include in results (default: 0.33)

    Returns:
        List of (file_path, combined_score, signal_breakdown) tuples,
        sorted by combined_score descending.
        Only files with combined_score >= threshold are returned.
    """
    # Get all file nodes
    file_nodes = [
        n for n in graph.nodes()
        if graph.nodes[n].get('type') == 'file' and n != target_file
    ]

    if not file_nodes:
        return []

    # Extract target file features
    target_code = graph.nodes[target_file].get('code', '')
    target_basename = os.path.basename(target_file)
    target_name_tokens = tokenize_filename(target_basename)
    target_struct = extract_structural_features(target_code)
    target_identifiers = extract_identifiers(target_code)

    # Extract features for all candidate files
    candidates = []
    corpus_identifiers = []

    for file_path in file_nodes:
        code = graph.nodes[file_path].get('code', '')
        basename = os.path.basename(file_path)

        candidates.append({
            'path': file_path,
            'name_tokens': tokenize_filename(basename),
            'struct': extract_structural_features(code),
            'identifiers': extract_identifiers(code)
        })
        corpus_identifiers.append(candidates[-1]['identifiers'])

    # Build BM25 index
    bm25 = BM25(corpus_identifiers)
    bm25_scores = bm25.get_scores(target_identifiers)

    # Normalize BM25 scores to [0, 1]
    # Handle negative scores by shifting to non-negative first
    if bm25_scores:
        min_bm25 = min(bm25_scores)
        if min_bm25 < 0:
            # Shift all scores to be non-negative
            bm25_scores = [score - min_bm25 for score in bm25_scores]

        max_bm25 = max(bm25_scores)
        if max_bm25 == 0:
            bm25_scores_norm = [0.0] * len(bm25_scores)
        else:
            bm25_scores_norm = [score / max_bm25 for score in bm25_scores]
    else:
        bm25_scores_norm = []

    # Compute combined scores
    results = []
    for idx, candidate in enumerate(candidates):
        # Signal 1: File name overlap
        signal1 = overlap_coefficient(target_name_tokens, candidate['name_tokens'])

        # Signal 2: Structural similarity
        signal2 = structural_similarity(target_struct, candidate['struct'])

        # Signal 3: BM25 on identifiers
        signal3 = bm25_scores_norm[idx]

        # Combined score (equal weights)
        combined = (signal1 + signal2 + signal3) / 3.0

        results.append((
            candidate['path'],
            combined,
            {
                'name_overlap': signal1,
                'structural': signal2,
                'identifier_bm25': signal3
            }
        ))

    # Sort by combined score descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Filter by threshold and return top_k
    filtered_results = [r for r in results if r[1] >= threshold]

    return filtered_results[:top_k]
