"""
Data Preprocessor for European Insurance Federated Learning
Handles preprocessing, feature engineering, and scaling
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from dataclasses import dataclass, field

from src.core.loader.data_schema import DataSchema, get_insurance_schema, ColumnType


@dataclass
class PreprocessorConfig:
    """Configuration for data preprocessing"""
    handle_missing: bool = True
    missing_strategy: str = 'mean'  # 'mean', 'median', 'mode', 'constant'
    missing_value: Optional[float] = None
    scale_numerical: bool = True
    scaling_method: str = 'standard'  # 'standard', 'minmax', 'robust'
    encode_categorical: bool = True
    encoding_method: str = 'onehot'  # 'onehot', 'label'
    drop_metadata: bool = True
    target_encoding: bool = False
    custom_transformations: Dict = field(default_factory=dict)


class DataPreprocessor:
    """
    Data Preprocessor for Insurance Dataset
    Handles preprocessing, feature engineering, and scaling
    """

    def __init__(self, schema: Optional[DataSchema] = None, config: Optional[PreprocessorConfig] = None):
        """
        Initialize DataPreprocessor

        Parameters:
        -----------
        schema : Optional[DataSchema]
            Data schema (default: uses insurance schema)
        config : Optional[PreprocessorConfig]
            Preprocessor configuration
        """
        self.schema = schema or get_insurance_schema()
        self.config = config or PreprocessorConfig()

        # Preprocessing objects
        self.numerical_imputer = None
        self.categorical_imputer = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = []
        self.is_fitted = False

        # Store fitted parameters
        self.numerical_columns = self.schema.get_numerical_columns()
        self.categorical_columns = self.schema.get_categorical_columns()
        self.target_columns = self.schema.get_target_columns()
        self.metadata_columns = self.schema.get_metadata_columns()
        self.id_columns = self.schema.get_id_columns()

    def fit(self, df: pd.DataFrame):
        """
        Fit preprocessor on training data

        Parameters:
        -----------
        df : pd.DataFrame
            Training data to fit on
        """
        # Separate features and targets
        X = df.drop(columns=self.target_columns, errors='ignore')
        y = df[self.target_columns] if all(col in df.columns for col in self.target_columns) else None

        # Identify columns to preprocess
        numeric_cols = [col for col in self.numerical_columns if col in X.columns]
        categorical_cols = [col for col in self.categorical_columns if col in X.columns]
        metadata_cols = [col for col in self.metadata_columns if col in X.columns]

        # Numerical imputation
        if self.config.handle_missing and numeric_cols:
            self.numerical_imputer = SimpleImputer(
                strategy=self.config.missing_strategy,
                fill_value=self.config.missing_value
            )
            self.numerical_imputer.fit(X[numeric_cols])

        # Categorical imputation
        if self.config.handle_missing and categorical_cols:
            self.categorical_imputer = SimpleImputer(
                strategy='most_frequent'
            )
            self.categorical_imputer.fit(X[categorical_cols])

        # Scaling
        if self.config.scale_numerical and numeric_cols:
            if self.config.scaling_method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = StandardScaler()  # Default to standard
            self.scaler.fit(X[numeric_cols])

        # Categorical encoding
        if self.config.encode_categorical and categorical_cols:
            for col in categorical_cols:
                if self.config.encoding_method == 'onehot':
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                else:
                    encoder = LabelEncoder()
                self.encoders[col] = encoder
                # Fit encoder
                encoder.fit(X[col].fillna('missing'))

        # Store feature names for transformation
        self.feature_names = self._get_feature_names(X)
        self.is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor

        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform

        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        # Make a copy
        data = df.copy()

        # Separate features and targets
        X = data.drop(columns=self.target_columns, errors='ignore')
        y = data[self.target_columns] if all(col in data.columns for col in self.target_columns) else None

        # Identify columns
        numeric_cols = [col for col in self.numerical_columns if col in X.columns]
        categorical_cols = [col for col in self.categorical_columns if col in X.columns]
        metadata_cols = [col for col in self.metadata_columns if col in X.columns]
        id_cols = [col for col in self.id_columns if col in X.columns]

        # Process numerical features
        if numeric_cols and self.numerical_imputer is not None:
            X[numeric_cols] = self.numerical_imputer.transform(X[numeric_cols])

        if numeric_cols and self.scaler is not None:
            X[numeric_cols] = self.scaler.transform(X[numeric_cols])

        # Process categorical features
        if categorical_cols and self.categorical_imputer is not None:
            X[categorical_cols] = self.categorical_imputer.transform(X[categorical_cols])

        if categorical_cols and self.encoders:
            encoded_dfs = []
            for col in categorical_cols:
                if col in self.encoders:
                    encoder = self.encoders[col]
                    if self.config.encoding_method == 'onehot':
                        encoded = encoder.transform(X[col].fillna('missing'))
                        # Create column names
                        feature_names = encoder.get_feature_names_out([col])
                        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                        encoded_dfs.append(encoded_df)
                    else:
                        X[col] = encoder.transform(X[col].fillna('missing'))

            if self.config.encoding_method == 'onehot':
                # Drop original categorical columns and add encoded ones
                X = X.drop(columns=categorical_cols)
                X = pd.concat([X] + encoded_dfs, axis=1)

        # Drop metadata columns if configured
        if self.config.drop_metadata and metadata_cols:
            X = X.drop(columns=metadata_cols, errors='ignore')

        # Keep ID columns
        if id_cols:
            # Keep IDs for reference
            pass

        # Add back target columns
        if y is not None:
            X = pd.concat([X, y], axis=1)

        return X

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform data

        Parameters:
        -----------
        df : pd.DataFrame
            Data to fit and transform

        Returns:
        --------
        pd.DataFrame
            Transformed data
        """
        self.fit(df)
        return self.transform(df)

    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names after transformation"""
        features = []

        # Numerical features
        numeric_cols = [col for col in self.numerical_columns if col in X.columns]
        features.extend(numeric_cols)

        # Categorical features (encoded)
        categorical_cols = [col for col in self.categorical_columns if col in X.columns]
        if self.config.encode_categorical and categorical_cols:
            if self.config.encoding_method == 'onehot':
                for col in categorical_cols:
                    if col in self.encoders:
                        encoder = self.encoders[col]
                        features.extend(encoder.get_feature_names_out([col]))
            else:
                features.extend(categorical_cols)

        # Metadata features (if not dropped)
        if not self.config.drop_metadata:
            metadata_cols = [col for col in self.metadata_columns if col in X.columns]
            features.extend(metadata_cols)

        return features

    def get_feature_names(self) -> List[str]:
        """Get transformed feature names"""
        return self.feature_names

    def inverse_transform_numerical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform numerical features"""
        if self.scaler is None:
            return X

        X_transformed = X.copy()
        numeric_cols = [col for col in self.numerical_columns if col in X.columns]
        if numeric_cols:
            X_transformed[numeric_cols] = self.scaler.inverse_transform(X[numeric_cols])

        return X_transformed

    def get_metadata(self) -> Dict:
        """Get preprocessor metadata"""
        return {
            'is_fitted': self.is_fitted,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'target_columns': self.target_columns,
            'feature_names': self.feature_names,
            'config': {
                'handle_missing': self.config.handle_missing,
                'missing_strategy': self.config.missing_strategy,
                'scale_numerical': self.config.scale_numerical,
                'scaling_method': self.config.scaling_method,
                'encode_categorical': self.config.encode_categorical,
                'encoding_method': self.config.encoding_method,
            }
        }