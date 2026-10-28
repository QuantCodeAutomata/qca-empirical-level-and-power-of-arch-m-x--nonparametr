import numpy as np
from typing import Set


def harmonic_number(d: int) -> float:
    """Compute C_d = sum_{l=1}^d 1/l."""
    return sum(1.0 / l for l in range(1, d + 1))


def by_fdr(pvalues: np.ndarray, q: float = 0.05) -> Set[int]:
    """
    Benjamini-Yekutieli FDR correction.
    Threshold c_l = (l/d) * q / C_d. Returns set of rejected indices (0-based).
    # Custom -- Context7 found no library equivalent
    """
    d = len(pvalues)
    C_d = harmonic_number(d)
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    thresholds = np.arange(1, d + 1) / (d * C_d) * q
    eligible = np.where(sorted_p <= thresholds)[0]
    if len(eligible) == 0:
        return set()
    k = eligible[-1]
    return set(sorted_idx[:k + 1].tolist())


def compute_selection_metrics(selected: Set[int], active_set: Set[int],
                               inactive_set: Set[int]) -> dict:
    """Compute CS, IS, CE, IE selection metrics."""
    all_vars = active_set | inactive_set
    not_selected = all_vars - selected
    return {
        'CS': len(selected & active_set),
        'IS': len(selected & inactive_set),
        'CE': len(not_selected & inactive_set),
        'IE': len(not_selected & active_set),
    }
