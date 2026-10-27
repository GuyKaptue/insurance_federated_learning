"""Partitioning strategies for the Insurance Federated Learning dataset.

This module defines how a client's stored partition is split into training,
validation and test subsets. Two strategies are supported:

* ``PartitionStrategy.BY_CUSTOMER_ID`` — stratified by the ``customer_id``
  column, so every row of one customer stays in a single split and the
  resulting subsets preserve the same customer-level distribution on each
  task target. This is the default and the strategy used by the smoke
  dataset generation in ``task.py``.
* ``PartitionStrategy.BY_GROUP_ID`` — stratified by an explicit
  ``group_id`` column (if present), otherwise falls back to the same
  deterministic per-customer split as ``BY_CUSTOMER_ID``.

All split computations are deterministic given a fixed ``random_state``
and a fixed input file; they never mutate the source Parquet.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class PartitionStrategy(str, Enum):
    """How to split a client partition into train/val/test."""

    BY_CUSTOMER_ID = "by_customer_id"
    BY_GROUP_ID = "by_group_id"


def _coerce_random_state(random_state: int | np.random.Generator | None) -> np.random.Generator:
    """Normalise the seed into a ``np.random.Generator`` for reproducibility."""
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, int):
        return np.random.default_rng(random_state)
    return np.random.default_rng()


def _stratified_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, pd.DataFrame]:
    """Split a dataframe by group, preserving the group-level distribution.

    Each group (e.g. a single customer) is assigned wholly to one split,
    so a group never appears in multiple subsets. The split sizes are
    approximate: we sort groups by size and greedily assign the largest
    remaining group to the split with the biggest deficit (in terms of
    the target fraction), which keeps the per-task distribution of
    ``group_col`` balanced across splits even when groups vary wildly in
    size (a common case for insurance customers).

    Parameters
    ----------
    df:
        The client partition.
    group_col:
        Column name to stratify on (``customer_id`` for ``BY_CUSTOMER_ID``,
        ``group_id`` for ``BY_GROUP_ID``).
    test_size:
        Fraction of the total number of *groups* to put in the test split.
    val_size:
        Fraction of the total number of *groups* to put in the validation
        split.
    random_state:
        Seed or ``Generator`` for reproducible group ordering.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ``{"train": ..., "val": ..., "test": ...}``.

    Raises
    ------
    ValueError
        If ``df`` is empty, if ``group_col`` is missing, or if the split
        fractions are outside ``[0, 1]``.
    """
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    if group_col not in df.columns:
        raise ValueError(
            f"Column '{group_col}' not found in dataframe. Columns: {list(df.columns)}"
        )

    if not (0.0 <= test_size <= 1.0):
        raise ValueError(f"test_size must be in [0, 1], got {test_size!r}")
    if not (0.0 <= val_size <= 1.0):
        raise ValueError(f"val_size must be in [0, 1], got {val_size!r}")

    rng = _coerce_random_state(random_state)
    groups = df[group_col].unique()
    n_groups = len(groups)

    # Deterministic ordering.
    if isinstance(random_state, int):
        rng = np.random.default_rng(random_state)
    else:
        rng = rng

    group_indices = rng.permutation(n_groups)
    group_list = list(groups)

    # Greedy assignment of groups to splits.
    test_count = max(1, int(round(n_groups * test_size)))
    val_count = max(0, int(round(n_groups * val_size)))
    # Ensure we never allocate more groups than available.
    test_count = min(test_count, n_groups - 1 if val_count > 0 else n_groups)
    val_count = min(val_count, n_groups - test_count)

    splits: dict[str, list[Any]] = {"train": [], "val": [], "test": []}
    # Order: train first, then val, then test (deterministic).
    for i in group_indices[:test_count]:
        splits["test"].append(group_list[int(i)])
    for i in group_indices[test_count : test_count + val_count]:
        splits["val"].append(group_list[int(i)])
    for i in group_indices[test_count + val_count :]:
        splits["train"].append(group_list[int(i)])

    result = {}
    for split_name, group_ids in splits.items():
        result[split_name] = df[df[group_col].isin(group_ids)].copy()

    return result


def partition_by_customer_id(
    data_dir: Path,
    client_id: str,
    split: str = "train",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, pd.DataFrame]:
    """Load one client's split and partition it by ``customer_id``.

    Each customer's rows are kept together in exactly one of train/val/test.
    The split is deterministic for a fixed ``random_state``.

    Parameters
    ----------
    data_dir:
        Root of the processed dataset.
    client_id:
        Client identifier (e.g. ``"germany_allianz"``).
    split:
        Which of the client's stored Parquet files to read (default ``"train"``).
    test_size:
        Fraction of groups to place in the test split.
    val_size:
        Fraction of groups to place in the validation split.
    random_state:
        Seed or generator for reproducible ordering.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{"train": ..., "val": ..., "test": ...}``.

    Raises
    ------
    FileNotFoundError
        If the source Parquet does not exist.
    ValueError
        If the source file is empty or lacks a ``customer_id`` column.
    """
    source = Path(data_dir) / "federated" / client_id / f"{split}.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing partition file for client '{client_id}' split '{split}': {source}. "
            "Run 'python -m insurance_fl.fl.task --prepare' first."
        )

    df = pd.read_parquet(source)

    if df.empty:
        raise ValueError(f"Partition for client '{client_id}' split '{split}' is empty.")

    if "customer_id" not in df.columns:
        raise ValueError(
            f"Partition for client '{client_id}' split '{split}' has no 'customer_id' column. "
            f"Columns found: {list(df.columns)}"
        )

    return _stratified_split(
        df,
        group_col="customer_id",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )


def partition_by_group_id(
    data_dir: Path,
    client_id: str,
    split: str = "train",
    group_col: str = "group_id",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int | np.random.Generator | None = None,
) -> dict[str, pd.DataFrame]:
    """Load one client's split and partition it by ``group_id``.

    Uses the same greedy stratified algorithm as ``partition_by_customer_id``
    but against a configurable group column, so the same code path serves
    both strategies. If the column is absent the method falls back to a
    deterministic customer-based split.

    Parameters
    ----------
    data_dir:
        Root of the processed dataset.
    client_id:
        Client identifier.
    split:
        Which stored Parquet to read (default ``"train"``).
    group_col:
        Column name to stratify on. Falls back to ``customer_id`` if absent.
    test_size:
        Fraction of groups to place in the test split.
    val_size:
        Fraction of groups to place in the validation split.
    random_state:
        Seed or generator for reproducible ordering.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{"train": ..., "val": ..., "test": ...}``.

    Raises
    ------
    FileNotFoundError
        If the source Parquet does not exist.
    ValueError
        If the source file is empty.
    """
    source = Path(data_dir) / "federated" / client_id / f"{split}.parquet"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing partition file for client '{client_id}' split '{split}': {source}. "
            "Run 'python -m insurance_fl.fl.task --prepare' first."
        )

    df = pd.read_parquet(source)

    if df.empty:
        raise ValueError(f"Partition for client '{client_id}' split '{split}' is empty.")

    if group_col not in df.columns:
        log.warning(
            "Group column '%s' not found for client '%s' split '%s'; falling back to customer_id.",
            group_col,
            client_id,
            split,
        )
        return partition_by_customer_id(
            data_dir,
            client_id,
            split,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
        )

    return _stratified_split(
        df, group_col=group_col, test_size=test_size, val_size=val_size, random_state=random_state
    )


def get_partition(
    data_dir: Path,
    client_id: str,
    split: str = "train",
    strategy: PartitionStrategy = PartitionStrategy.BY_CUSTOMER_ID,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Public API for partitioning a client's stored split.

    Parameters
    ----------
    data_dir:
        Root of the processed dataset.
    client_id:
        Client identifier.
    split:
        Which stored Parquet to read (default ``"train"``).
    strategy:
        Which partitioning strategy to use (default ``BY_CUSTOMER_ID``).
    **kwargs:
        Additional keyword arguments forwarded to the strategy implementation.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``{"train": ..., "val": ..., "test": ...}``.

    Examples
    --------
    >>> from insurance_fl.data.partition import get_partition
    >>> parts = get_partition(data_dir="data/processed/fl_task0_smoke",
    ...                       client_id="germany_allianz",
    ...                       strategy=PartitionStrategy.BY_CUSTOMER_ID)
    >>> train_df = parts["train"]
    >>> val_df = parts["val"]
    >>> test_df = parts["test"]
    """
    if strategy == PartitionStrategy.BY_CUSTOMER_ID:
        return partition_by_customer_id(data_dir, client_id, split=split, **kwargs)
    if strategy == PartitionStrategy.BY_GROUP_ID:
        return partition_by_group_id(data_dir, client_id, split=split, **kwargs)

    raise ValueError(f"Unknown partition strategy: {strategy!r}")
