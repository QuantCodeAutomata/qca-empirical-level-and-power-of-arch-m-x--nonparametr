import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.selection import harmonic_number, by_fdr, compute_selection_metrics


def test_harmonic_number():
    assert abs(harmonic_number(1) - 1.0) < 1e-10
    assert abs(harmonic_number(2) - 1.5) < 1e-10
    assert abs(harmonic_number(5) - (1 + 0.5 + 1/3 + 0.25 + 0.2)) < 1e-10


def test_by_fdr_all_null():
    # All p-values = 1, no rejection
    pvals = np.ones(5)
    selected = by_fdr(pvals, q=0.05)
    assert len(selected) == 0


def test_by_fdr_all_significant():
    # All p-values very small, all rejected
    pvals = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10])
    selected = by_fdr(pvals, q=0.05)
    assert len(selected) == 5


def test_by_fdr_partial_rejection():
    pvals = np.array([0.001, 0.003, 0.05, 0.3, 0.9])
    selected = by_fdr(pvals, q=0.05)
    # First two should be selected, others not
    assert 0 in selected
    assert 1 in selected
    assert 4 not in selected


def test_by_fdr_correct_thresholds_d5():
    # BY thresholds for d=5, q=0.05: C_5 = 2.28333
    # c_1 = 1/(5*2.28333)*0.05 = 0.004384
    d = 5
    C5 = harmonic_number(d)
    c1 = 1/(d * C5) * 0.05
    assert abs(c1 - 0.004383) < 1e-5


def test_by_fdr_pvalue_between_0_and_1():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pvals = rng.uniform(0, 1, 10)
        selected = by_fdr(pvals, q=0.05)
        assert all(0 <= i < 10 for i in selected)


def test_selection_metrics_perfect():
    selected = {0, 2, 3}
    active = {0, 2, 3}
    inactive = {1, 4}
    m = compute_selection_metrics(selected, active, inactive)
    assert m['CS'] == 3
    assert m['IS'] == 0
    assert m['CE'] == 2
    assert m['IE'] == 0


def test_selection_metrics_all_missed():
    selected = set()
    active = {0, 2, 3}
    inactive = {1, 4}
    m = compute_selection_metrics(selected, active, inactive)
    assert m['CS'] == 0
    assert m['IS'] == 0
    assert m['CE'] == 2
    assert m['IE'] == 3


def test_selection_metrics_all_selected():
    selected = {0, 1, 2, 3, 4}
    active = {0, 2, 3}
    inactive = {1, 4}
    m = compute_selection_metrics(selected, active, inactive)
    assert m['CS'] == 3
    assert m['IS'] == 2
    assert m['CE'] == 0
    assert m['IE'] == 0
