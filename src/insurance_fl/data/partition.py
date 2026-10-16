"""
Data partitioning utilities for federated learning.

Provides non-IID partitioning using Dirichlet distributions and stratified
splits. Every index is assigned exactly once — no sample is lost or duplicated.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def dirichlet_partition(
    indices: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples_per_client: int = 1,
) -> List[np.ndarray]:
    """
    Partition indices into non-IID client shards using a Dirichlet distribution.

    Each client receives a proportion of samples from each class drawn from
    Dirichlet(alpha). Lower alpha = more skewed (more non-IID).

    Parameters
    ----------
    indices : np.ndarray
        Array of sample indices to partition.
    labels : np.ndarray
        Array of class labels corresponding to indices.
    num_clients : int
        Number of clients.
    alpha : float
        Dirichlet concentration parameter. Lower = more heterogeneous.
    seed : int
        Random seed for reproducibility.
    min_samples_per_client : int
        Minimum samples guaranteed per client.

    Returns
    -------
    List[np.ndarray]
        List of index arrays, one per client. Every input index appears
        exactly once across all clients.

    Raises
    ------
    ValueError
        If alpha <= 0 or num_clients < 1.
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}")
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if len(indices) == 0:
        return [np.array([], dtype=int) for _ in range(num_clients)]

    rng = np.random.default_rng(seed)
    indices = np.asarray(indices)
    labels = np.asarray(labels)

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Initialize empty partitions
    partitions: List[List[int]] = [[] for _ in range(num_clients)]

    # For each class, distribute samples according to Dirichlet(alpha)
    for label in unique_labels:
        class_indices = indices[labels == label]
        n_class = len(class_indices)

        if n_class == 0:
            continue

        # Draw proportions from Dirichlet
        proportions = rng.dirichlet([alpha] * num_clients)

        # Compute number of samples per client for this class
        # Use cumulative proportion method to ensure every index assigned
        cum_proportions = np.cumsum(proportions)
        cum_proportions = cum_proportions / cum_proportions[-1]  # normalize

        split_points = (cum_proportions * n_class).astype(int)
        split_points = np.clip(split_points, 0, n_class)

        # Ensure monotonic non-decreasing split points
        split_points = np.maximum.accumulate(split_points)

        # Shuffle class indices before splitting
        shuffled = rng.permutation(class_indices)

        # Assign splits
        prev = 0
        for client_id in range(num_clients):
            end = split_points[client_id]
            client_indices = shuffled[prev:end]
            partitions[client_id].extend(client_indices.tolist())
            prev = end

        # Assign any remaining samples to the last client
        if prev < n_class:
            partitions[-1].extend(shuffled[prev:].tolist())

    # Convert to arrays and ensure every index is assigned exactly once
    result: List[np.ndarray] = []
    assigned_list: List[int] = []
    for client_id in range(num_clients):
        client_arr = np.array(partitions[client_id], dtype=int)
        result.append(client_arr)
        assigned_list.extend(client_arr.tolist())

    # Verify every index assigned exactly once
    all_assigned = np.sort(np.array(assigned_list, dtype=int))
    expected = np.sort(indices)
    if not np.array_equal(all_assigned, expected):
        raise ValueError(
            "Partitioning error: not every index assigned exactly once. "
            f"Assigned {len(all_assigned)}, expected {len(expected)}"
        )

    # Ensure min samples per client: keep transferring from surplus donors
    # until every client meets the minimum (a single transfer + break was
    # insufficient when a client started far below the minimum).
    for client_id in range(num_clients):
        while len(result[client_id]) < min_samples_per_client:
            donor_id = next(
                (
                    d
                    for d in range(num_clients)
                    if d != client_id and len(result[d]) > min_samples_per_client
                ),
                None,
            )
            if donor_id is None:  # not enough surplus anywhere
                break
            result[client_id] = np.append(result[client_id], result[donor_id][-1])
            result[donor_id] = result[donor_id][:-1]

    return result


def iid_partition(
    indices: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Partition indices into IID client shards (equal random split).

    Parameters
    ----------
    indices : np.ndarray
        Array of sample indices to partition.
    num_clients : int
        Number of clients.
    seed : int
        Random seed.

    Returns
    -------
    List[np.ndarray]
        List of index arrays, one per client.
    """
    if num_clients < 1:
        raise ValueError(f"num_clients must be >= 1, got {num_clients}")
    if len(indices) == 0:
        return [np.array([], dtype=int) for _ in range(num_clients)]

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    return np.array_split(shuffled, num_clients)


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    stratify_col: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into train/test with stratification."""
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[stratify_col]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
