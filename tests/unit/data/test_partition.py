"""Unit tests for the data partition strategies (src/insurance_fl/data/partition.py).

Tests the customer-group stratified splitting used to split a client
partition into train/val/test subsets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insurance_fl.data.partition import (
    PartitionStrategy,
    get_partition,
    partition_by_customer_id,
    partition_by_group_id,
)


@pytest.fixture
def group_partition_df() -> pd.DataFrame:
    """A partition with explicit group_id column."""
    np.random.seed(42)
    data = []
    for gid, n in [("group_0", 10), ("group_1", 5), ("group_2", 8)]:
        for i in range(n):
            data.append({"group_id": gid, "x": i, "value": np.random.uniform(0, 100)})
    return pd.DataFrame(data)


@pytest.fixture
def small_partition_df() -> pd.DataFrame:
    """A small partition with 4 customers of varying sizes."""
    np.random.seed(42)
    data = []
    # Customer A: 10 rows
    for i in range(10):
        data.append(
            {
                "customer_id": "cust_A",
                "age": np.random.uniform(18, 90),
                "annual_income": np.random.uniform(20000, 150000),
            }
        )
    # Customer B: 3 rows
    for i in range(3):
        data.append(
            {
                "customer_id": "cust_B",
                "age": np.random.uniform(18, 90),
                "annual_income": np.random.uniform(20000, 150000),
            }
        )
    # Customer C: 7 rows
    for i in range(7):
        data.append(
            {
                "customer_id": "cust_C",
                "age": np.random.uniform(18, 90),
                "annual_income": np.random.uniform(20000, 150000),
            }
        )
    # Customer D: 2 rows
    for i in range(2):
        data.append(
            {
                "customer_id": "cust_D",
                "age": np.random.uniform(18, 90),
                "annual_income": np.random.uniform(20000, 150000),
            }
        )
    return pd.DataFrame(data)


class TestPartitionByCustomerId:
    """Tests for partition_by_customer_id()."""

    def test_partition_creates_three_subsets(
        self, small_partition_df: pd.DataFrame, tmp_path
    ) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        small_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = partition_by_customer_id(data_dir, "germany_allianz", "train")
        assert set(parts.keys()) == {"train", "val", "test"}

    def test_all_rows_assigned_exactly_once(
        self, small_partition_df: pd.DataFrame, tmp_path
    ) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        small_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = partition_by_customer_id(data_dir, "germany_allianz", "train")
        assert sum(len(df) for df in parts.values()) == len(small_partition_df)

    def test_customer_rows_not_split_across_subsets(
        self, small_partition_df: pd.DataFrame, tmp_path
    ) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        small_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = partition_by_customer_id(data_dir, "germany_allianz", "train")
        cust_a_rows = [df[df["customer_id"] == "cust_A"] for df in parts.values()]
        cust_a_in_split = sum(1 for rows in cust_a_rows if len(rows) > 0)
        assert cust_a_in_split == 1, "cust_A should appear in exactly one split"
        cust_a_total = sum(len(rows) for rows in cust_a_rows)
        assert cust_a_total == 10, f"cust_A rows split across subsets: {cust_a_total}"

    def test_deterministic_with_seed(self, small_partition_df: pd.DataFrame, tmp_path) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        small_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts1 = partition_by_customer_id(data_dir, "germany_allianz", "train", random_state=42)
        parts2 = partition_by_customer_id(data_dir, "germany_allianz", "train", random_state=42)
        for split_name in parts1:
            pd.testing.assert_frame_equal(parts1[split_name], parts2[split_name])

    def test_different_seed_different_splits(
        self, small_partition_df: pd.DataFrame, tmp_path
    ) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        small_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts1 = partition_by_customer_id(data_dir, "germany_allianz", "train", random_state=1)
        parts2 = partition_by_customer_id(data_dir, "germany_allianz", "train", random_state=2)
        different = any(not df1.equals(df2) for df1, df2 in zip(parts1.values(), parts2.values()))
        assert different, "Different seeds produced identical splits"

    def test_missing_customer_id_column_raises(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        df = pd.DataFrame({"age": [25], "annual_income": [50000]})
        df.to_parquet(str(client_dir / "train.parquet"), index=False)
        with pytest.raises(ValueError, match="customer_id"):
            partition_by_customer_id(data_dir, "germany_allianz", "train")

    def test_empty_partition_raises(self, tmp_path) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        pd.DataFrame({"customer_id": [], "age": []}).to_parquet(
            str(client_dir / "train.parquet"), index=False
        )
        with pytest.raises(ValueError, match="empty"):
            partition_by_customer_id(data_dir, "germany_allianz", "train")


class TestPartitionByGroupId:
    """Tests for partition_by_group_id()."""

    def test_partition_creates_three_subsets(
        self, group_partition_df: pd.DataFrame, tmp_path
    ) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        group_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = partition_by_group_id(data_dir, "germany_allianz", "train", group_col="group_id")
        assert set(parts.keys()) == {"train", "val", "test"}

    def test_all_rows_assigned_exactly_once(
        self, group_partition_df: pd.DataFrame, tmp_path
    ) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        group_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = partition_by_group_id(data_dir, "germany_allianz", "train", group_col="group_id")
        assert sum(len(df) for df in parts.values()) == len(group_partition_df)

    def test_group_rows_not_split_across_subsets(
        self, group_partition_df: pd.DataFrame, tmp_path
    ) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        group_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = partition_by_group_id(data_dir, "germany_allianz", "train", group_col="group_id")
        for split_name, df in parts.items():
            for group_id in df["group_id"].unique():
                group_rows = df[df["group_id"] == group_id]
                assert (
                    len(group_rows)
                    == group_partition_df[group_partition_df["group_id"] == group_id].shape[0]
                ), f"Group {group_id} split across {split_name}"


class TestGetPartition:
    """Tests for get_partition() public API."""

    def test_default_strategy_is_customer_id(
        self, small_partition_df: pd.DataFrame, tmp_path
    ) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        small_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = get_partition(data_dir, "germany_allianz", "train")
        assert set(parts.keys()) == {"train", "val", "test"}

    def test_with_by_customer_id_strategy(self, small_partition_df: pd.DataFrame, tmp_path) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        small_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = get_partition(
            data_dir, "germany_allianz", "train", strategy=PartitionStrategy.BY_CUSTOMER_ID
        )
        assert set(parts.keys()) == {"train", "val", "test"}

    def test_with_by_group_id_strategy(self, group_partition_df: pd.DataFrame, tmp_path) -> None:
        data_dir = tmp_path / "data"
        client_dir = data_dir / "federated" / "germany_allianz"
        client_dir.mkdir(parents=True)
        group_partition_df.to_parquet(str(client_dir / "train.parquet"), index=False)
        parts = get_partition(
            data_dir,
            "germany_allianz",
            "train",
            strategy=PartitionStrategy.BY_GROUP_ID,
            group_col="group_id",
        )
        assert set(parts.keys()) == {"train", "val", "test"}
