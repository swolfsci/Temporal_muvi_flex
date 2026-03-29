"""Tests for gene set preprocessing."""

import pytest
from tpacmon.tools.feature_sets import FeatureSet, FeatureSets
from tpacmon.tools.gene_set_prep import jaccard_similarity, merge_gene_sets, prepare_gene_sets


@pytest.fixture
def sample_gene_sets():
    gs1 = FeatureSet(features=[f"gene_{i}" for i in range(20)], name="set_A")
    gs2 = FeatureSet(features=[f"gene_{i}" for i in range(15, 35)], name="set_B")  # overlaps with A
    gs3 = FeatureSet(features=[f"gene_{i}" for i in range(100, 130)], name="set_C")  # disjoint
    gs4 = FeatureSet(features=[f"gene_{i}" for i in range(5)], name="set_tiny")  # too small
    return FeatureSets([gs1, gs2, gs3, gs4])


class TestJaccardSimilarity:
    def test_identical(self):
        a = frozenset(["a", "b", "c"])
        assert jaccard_similarity(a, a) == 1.0

    def test_disjoint(self):
        a = frozenset(["a", "b"])
        b = frozenset(["c", "d"])
        assert jaccard_similarity(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        # intersection=2, union=4
        assert jaccard_similarity(a, b) == 0.5

    def test_empty_sets(self):
        assert jaccard_similarity(frozenset(), frozenset()) == 1.0


class TestMergeGeneSets:
    def test_no_merge_when_disjoint(self, sample_gene_sets):
        # With high threshold, similar sets should merge but disjoint should stay
        result = merge_gene_sets(sample_gene_sets, threshold=0.1)
        # set_A and set_B have some overlap, set_C is disjoint
        assert len(result) >= 2

    def test_single_set(self):
        gs = FeatureSets([FeatureSet(features=["a", "b"], name="only")])
        result = merge_gene_sets(gs)
        assert len(result) == 1

    def test_identical_sets_merge(self):
        gs1 = FeatureSet(features=["a", "b", "c"], name="s1")
        gs2 = FeatureSet(features=["a", "b", "c"], name="s2")
        result = merge_gene_sets(FeatureSets([gs1, gs2]), threshold=0.9)
        assert len(result) == 1


class TestPrepareGeneSets:
    def test_size_filter(self, sample_gene_sets):
        features = [f"gene_{i}" for i in range(200)]
        result = prepare_gene_sets(sample_gene_sets, features, min_size=10, max_size=500)
        # set_tiny (5 genes) should be filtered out
        for gs in list(result.feature_sets):
            assert len(gs) >= 10

    def test_respects_available_features(self, sample_gene_sets):
        # Only provide genes 0-25 — set_C (100-130) should be empty after intersection
        features = [f"gene_{i}" for i in range(26)]
        result = prepare_gene_sets(sample_gene_sets, features, min_size=5, max_size=500)
        for gs in list(result.feature_sets):
            assert all(f in features for f in gs.features)

    def test_empty_result(self):
        gs = FeatureSets([FeatureSet(features=["x", "y"], name="tiny")])
        result = prepare_gene_sets(gs, ["a", "b", "c"], min_size=10)
        assert len(result) == 0
