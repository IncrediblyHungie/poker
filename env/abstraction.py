import numpy as np

def bucket_hand_vector(cards52: np.ndarray) -> int:
    """
    Map a 52‑length one‑hot hole+board encoding to an equity bucket id.
    Buckets = 1326 choose grouping by percentiles of hand strength vs random.
    """
    strength = _precomputed_table @ cards52     # dot product fast lookup
    return int(np.digitize(strength, _bucket_edges))
