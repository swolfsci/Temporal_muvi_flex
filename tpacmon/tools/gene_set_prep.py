"""Gene set preprocessing: size filtering + Jaccard similarity merging."""

import logging
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from tpacmon.tools.feature_sets import FeatureSet, FeatureSets

logger = logging.getLogger(__name__)


def jaccard_similarity(a: frozenset, b: frozenset) -> float:
    """Compute Jaccard similarity between two sets."""
    if len(a) == 0 and len(b) == 0:
        return 1.0
    return len(a & b) / len(a | b)


def merge_gene_sets(
    gene_sets: FeatureSets,
    threshold: float = 0.7,
    method: str = "ward",
) -> FeatureSets:
    """Merge gene sets with high Jaccard similarity using hierarchical clustering.

    Args:
        gene_sets: Input FeatureSets object.
        threshold: Jaccard similarity threshold for merging (0-1). Higher = more aggressive merging.
        method: Linkage method for hierarchical clustering.

    Returns:
        FeatureSets with merged sets.
    """
    sets = list(gene_sets.feature_sets)
    n = len(sets)

    if n <= 1:
        return gene_sets

    # Compute pairwise Jaccard distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_similarity(sets[i].features, sets[j].features)
            dist_matrix[i, j] = 1.0 - sim
            dist_matrix[j, i] = dist_matrix[i, j]

    # Hierarchical clustering
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=method)
    clusters = fcluster(Z, t=1.0 - threshold, criterion="distance")

    # Merge sets within each cluster (union of features)
    merged = []
    for c in np.unique(clusters):
        members = [sets[i] for i in range(n) if clusters[i] == c]
        if len(members) == 1:
            merged.append(members[0])
        else:
            combined_features = frozenset().union(*(m.features for m in members))
            names = [m.name for m in members]
            merged_name = "|".join(sorted(names)[:3])
            if len(names) > 3:
                merged_name += f"|+{len(names) - 3}"
            merged.append(FeatureSet(
                features=combined_features,
                name=merged_name,
                description=f"Merged from {len(names)} sets: {', '.join(names)}",
            ))
            logger.info(f"Merged {len(names)} gene sets -> '{merged_name}' ({len(combined_features)} features)")

    return FeatureSets(merged)


def prepare_gene_sets(
    gene_sets: FeatureSets,
    feature_names: list[str],
    min_size: int = 10,
    max_size: int = 500,
    jaccard_threshold: float = 0.7,
) -> FeatureSets:
    """Full gene set preparation pipeline: filter by size, then merge similar sets.

    Args:
        gene_sets: Raw input gene sets.
        feature_names: Features available in the dataset (for intersection).
        min_size: Minimum gene set size after intersection.
        max_size: Maximum gene set size after intersection.
        jaccard_threshold: Similarity threshold for merging.

    Returns:
        Cleaned FeatureSets ready for prior mask construction.
    """
    # Intersect with available features
    available = set(feature_names)
    filtered_sets = []
    for gs in list(gene_sets.feature_sets):
        overlap = gs.features & available
        if min_size <= len(overlap) <= max_size:
            filtered_sets.append(FeatureSet(
                features=overlap,
                name=gs.name,
                description=gs.description,
            ))

    logger.info(f"Size filter: {len(gene_sets)} -> {len(filtered_sets)} gene sets "
                f"(min={min_size}, max={max_size})")

    if len(filtered_sets) == 0:
        logger.warning("No gene sets passed size filter!")
        return FeatureSets([])

    result = FeatureSets(filtered_sets)

    # Merge similar sets
    result = merge_gene_sets(result, threshold=jaccard_threshold)
    logger.info(f"After merging: {len(result)} gene sets")

    return result
