"""
Data Loader for European Insurance Federated Learning
Handles loading data from various formats and sources
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import pyarrow as pa
import pyarrow.parquet as pq

from src.core.loader.data_schema import DataSchema, get_insurance_schema


@dataclass
class DatasetSplit:
    """Container for dataset splits"""
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __len__(self):
        return len(self.train) + len(self.val) + len(self.test)

    def get_stats(self) -> Dict:
        """Get dataset statistics"""
        return {
            'train_samples': len(self.train),
            'val_samples': len(self.val),
            'test_samples': len(self.test),
            'total_samples': len(self)
        }


class DataLoader:
    """
    Data Loader for European Insurance Federated Learning
    Supports loading federated and centralized datasets
    """

    def __init__(self, data_path: Union[str, Path], schema: Optional[DataSchema] = None):
        """
        Initialize DataLoader

        Parameters:
        -----------
        data_path : Union[str, Path]
            Path to the dataset directory
        schema : Optional[DataSchema]
            Data schema for validation (default: uses insurance schema)
        """
        self.data_path = Path(data_path)
        self.schema = schema or get_insurance_schema()
        self.metadata = None
        self.centralized_data = None
        self.federated_data = None

    def load_metadata(self) -> Dict:
        """Load dataset metadata"""
        metadata_path = self.data_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        return self.metadata

    def load_federated_data(self, clients: Optional[List[str]] = None) -> Dict[str, DatasetSplit]:
        """
        Load federated (client-specific) data

        Parameters:
        -----------
        clients : Optional[List[str]]
            List of client IDs to load (loads all if None)

        Returns:
        --------
        Dict[str, DatasetSplit]
            Dictionary mapping client_id to DatasetSplit
        """
        if self.metadata is None:
            self.load_metadata()

        federated_path = self.data_path / "federated"
        if not federated_path.exists():
            raise FileNotFoundError(f"Federated data not found at {federated_path}")

        # Get list of clients
        if clients is None:
            clients = self.metadata.get('clients', [])
            if not clients:
                # Discover clients from directory
                clients = [d.name for d in federated_path.iterdir() if d.is_dir()]

        # Load data for each client
        self.federated_data = {}
        for client_id in clients:
            client_dir = federated_path / client_id
            if not client_dir.exists():
                print(f"Warning: Client {client_id} not found, skipping...")
                continue

            splits = {}
            for split in ['train', 'val', 'test']:
                file_path = client_dir / f"{split}.parquet"
                if file_path.exists():
                    splits[split] = pd.read_parquet(file_path)
                else:
                    # Try NPY format
                    npy_path = client_dir / f"{split}.npy"
                    if npy_path.exists():
                        # Load NPY (requires column names)
                        cols_path = client_dir / f"{split}_columns.txt"
                        if cols_path.exists():
                            with open(cols_path, 'r') as f:
                                columns = [line.strip() for line in f.readlines()]
                            data = np.load(npy_path)
                            splits[split] = pd.DataFrame(data, columns=columns)
                        else:
                            splits[split] = pd.DataFrame(np.load(npy_path))

            # Validate split completeness
            if all(split in splits for split in ['train', 'val', 'test']):
                self.federated_data[client_id] = DatasetSplit(
                    train=splits['train'],
                    val=splits['val'],
                    test=splits['test']
                )
            else:
                print(f"Warning: Incomplete data for client {client_id}")

        return self.federated_data

    def load_centralized_data(self) -> Optional[DatasetSplit]:
        """
        Load centralized dataset

        Returns:
        --------
        Optional[DatasetSplit]
            Centralized dataset splits or None if not available
        """
        centralized_path = self.data_path / "centralized"
        if not centralized_path.exists():
            print("Centralized dataset not found")
            return None

        splits = {}
        for split in ['train', 'val', 'test']:
            file_path = centralized_path / f"{split}.parquet"
            if file_path.exists():
                splits[split] = pd.read_parquet(file_path)
            else:
                # Try NPY format
                npy_path = centralized_path / f"{split}.npy"
                if npy_path.exists():
                    splits[split] = pd.DataFrame(np.load(npy_path))

        if all(split in splits for split in ['train', 'val', 'test']):
            self.centralized_data = DatasetSplit(
                train=splits['train'],
                val=splits['val'],
                test=splits['test']
            )
            return self.centralized_data

        return None

    def load_all(self, load_centralized: bool = True) -> Tuple[Dict[str, DatasetSplit], Optional[DatasetSplit]]:
        """
        Load both federated and centralized datasets

        Parameters:
        -----------
        load_centralized : bool
            Whether to load centralized dataset

        Returns:
        --------
        Tuple[Dict[str, DatasetSplit], Optional[DatasetSplit]]
            (federated_data, centralized_data)
        """
        federated = self.load_federated_data()
        centralized = self.load_centralized_data() if load_centralized else None
        return federated, centralized

    def get_client_data(self, client_id: str) -> Optional[DatasetSplit]:
        """Get data for a specific client"""
        if self.federated_data is None:
            self.load_federated_data()
        return self.federated_data.get(client_id)

    def get_sample(self, client_id: Optional[str] = None, split: str = 'train', n_samples: int = 5) -> pd.DataFrame:
        """
        Get a sample of data

        Parameters:
        -----------
        client_id : Optional[str]
            Client ID (if None, uses centralized data)
        split : str
            'train', 'val', or 'test'
        n_samples : int
            Number of samples to return

        Returns:
        --------
        pd.DataFrame
            Sample of data
        """
        if client_id is not None:
            data = self.get_client_data(client_id)
            if data is None:
                raise ValueError(f"Client {client_id} not found")
            df = getattr(data, split)
        else:
            if self.centralized_data is None:
                self.load_centralized_data()
            if self.centralized_data is None:
                raise ValueError("Centralized data not available")
            df = getattr(self.centralized_data, split)

        return df.head(n_samples)

    def get_schema(self) -> DataSchema:
        """Get data schema"""
        return self.schema

    def validate_data(self, df: pd.DataFrame, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Validate data against schema

        Parameters:
        -----------
        df : pd.DataFrame
            Data to validate
        strict : bool
            Whether to strictly validate all columns

        Returns:
        --------
        Tuple[bool, List[str]]
            (is_valid, errors)
        """
        errors = []

        # Check required columns
        required_cols = self.schema.get_column_names()
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")

        # Check data types
        for col_meta in self.schema.columns:
            if col_meta.name in df.columns:
                col_data = df[col_meta.name]
                expected_type = col_meta.data_type

                # Check for null values if not nullable
                if not col_meta.is_nullable and col_data.isnull().any():
                    errors.append(f"Column {col_meta.name} has null values but is not nullable")

                # Check value ranges for numerical columns
                if col_meta.column_type.value == 'numerical' and col_meta.min_value is not None:
                    invalid = col_data[(col_data < col_meta.min_value) | (col_data > col_meta.max_value)]
                    if not invalid.empty:
                        errors.append(f"Column {col_meta.name} has values outside range [{col_meta.min_value}, {col_meta.max_value}]")

        is_valid = len(errors) == 0
        if not is_valid and not strict:
            print("Validation warnings:")
            for error in errors:
                print(f"  ⚠️ {error}")

        return is_valid, errors

    def get_summary(self) -> Dict:
        """Get complete dataset summary"""
        summary = {
            'data_path': str(self.data_path),
            'metadata': self.metadata,
            'federated_clients': len(self.federated_data) if self.federated_data else 0,
            'centralized_available': self.centralized_data is not None
        }

        if self.federated_data:
            client_stats = {}
            for client_id, data in self.federated_data.items():
                client_stats[client_id] = data.get_stats()
            summary['federated_stats'] = client_stats

        if self.centralized_data:
            summary['centralized_stats'] = self.centralized_data.get_stats()

        return summary