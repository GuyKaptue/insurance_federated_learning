"""Leakage-safe tabular preprocessing (fit on train only).

Imputation, scaling and categorical encoding are fitted on the training
split and then applied unchanged to val/test/site. Targets are never used
to fit any transformer (only to attach alongside after transformation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler


@dataclass
class PreprocessorConfig:
    """Configuration for :class:`DataPreprocessor`."""

    handle_missing: bool = True
    missing_strategy: str = "mean"  # mean | median | most_frequent
    scale_numerical: bool = True
    scaling_method: str = "standard"  # standard | minmax
    encode_categorical: bool = True
    encoding_method: str = "onehot"  # onehot | label
    drop_metadata: bool = True
    target_columns: List[str] = field(default_factory=list)
    metadata_columns: List[str] = field(
        default_factory=lambda: [
            "customer_id",
            "country",
            "insurance_company",
            "regulatory_framework",
            "currency",
            "client_id",
        ]
    )
    id_columns: List[str] = field(default_factory=lambda: ["customer_id"])


class DataPreprocessor:
    """Impute / scale / encode insurance tabular data.

    Usage::

        pp = DataPreprocessor(PreprocessorConfig(target_columns=TARGETS))
        pp.fit(train_df)                     # fit on train ONLY
        X = pp.transform(val_df)             # apply unchanged
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        self.config = config or PreprocessorConfig()
        self.numerical_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.feature_cols_: List[str] = []
        self.imputer_: Optional[SimpleImputer] = None
        self.cat_imputer_: Optional[SimpleImputer] = None
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.encoders_: Dict[str, LabelEncoder] = {}
        self.onehot_: Optional[OneHotEncoder] = None
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    def _split_columns(self, df: pd.DataFrame) -> None:
        """Classify columns into numerical / categorical / excluded."""
        cfg = self.config
        excluded = set(cfg.target_columns) | set(cfg.metadata_columns) | set(cfg.id_columns)
        self.numerical_cols_ = [
            c for c in df.select_dtypes(include=["number"]).columns if c not in excluded
        ]
        self.categorical_cols_ = [
            c for c in df.select_dtypes(exclude=["number"]).columns if c not in excluded
        ]

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit imputers, scaler and encoders on the training data ONLY."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pandas DataFrame, got {type(df).__name__}")
        if len(df) == 0:
            raise ValueError("Cannot fit preprocessor on an empty DataFrame")

        self._split_columns(df)
        cfg = self.config

        if cfg.handle_missing and self.numerical_cols_:
            strategy = "constant" if cfg.missing_strategy == "constant" else cfg.missing_strategy
            kwargs = {"fill_value": 0.0} if strategy == "constant" else {}
            self.imputer_ = SimpleImputer(strategy=strategy, **kwargs)
            self.imputer_.fit(df[self.numerical_cols_])

        if cfg.handle_missing and self.categorical_cols_:
            self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
            self.cat_imputer_.fit(df[self.categorical_cols_].astype("object"))

        if cfg.scale_numerical and self.numerical_cols_:
            if cfg.scaling_method == "minmax":
                self.scaler_ = MinMaxScaler()
            elif cfg.scaling_method == "standard":
                self.scaler_ = StandardScaler()
            else:
                raise ValueError(f"Unknown scaling_method: {cfg.scaling_method!r}")
            scaled_input = (
                pd.DataFrame(
                    self.imputer_.transform(df[self.numerical_cols_]),
                    columns=self.numerical_cols_,
                    index=df.index,
                )
                if self.imputer_ is not None
                else df[self.numerical_cols_].astype("float64")
            )
            self.scaler_.fit(scaled_input)

        if cfg.encode_categorical and self.categorical_cols_:
            if cfg.encoding_method == "label":
                for col in self.categorical_cols_:
                    le = LabelEncoder()
                    values = df[col].astype("object")
                    values = values.fillna("__MISSING__").astype(str)
                    le.fit(values)
                    self.encoders_[col] = le
            elif cfg.encoding_method == "onehot":
                cat_block = df[self.categorical_cols_].astype("object").fillna("__MISSING__")
                self.onehot_ = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=np.float64
                )
                self.onehot_.fit(cat_block.astype(str))
            else:
                raise ValueError(f"Unknown encoding_method: {cfg.encoding_method!r}")

        self.feature_names_ = self._compute_feature_names()
        self.feature_cols_ = list(self.numerical_cols_) + list(self.categorical_cols_)
        self.is_fitted = True
        return self

    def _compute_feature_names(self) -> List[str]:
        """Return output feature names (numerical + encoded columns)."""
        names = list(self.numerical_cols_)
        if self.config.encode_categorical and self.categorical_cols_:
            if self.config.encoding_method == "onehot" and self.onehot_ is not None:
                names.extend(
                    [str(n) for n in self.onehot_.get_feature_names_out(self.categorical_cols_)]
                )
            else:
                names.extend(self.categorical_cols_)
        return [str(n) for n in names]

    def transform(self, df: pd.DataFrame, targets: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply fitted transformations; optionally re-attach target columns.

        Raises
        ------
        RuntimeError
            If :meth:`fit` was never called.
        ValueError
            If required feature columns are missing from ``df``.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform()")
        missing = [c for c in self.feature_cols_ if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns in transform data: {missing}")

        cfg = self.config
        parts: List[pd.DataFrame] = []

        if self.numerical_cols_:
            block = df[self.numerical_cols_].astype("float64")
            if self.imputer_ is not None:
                block = pd.DataFrame(
                    self.imputer_.transform(block),
                    columns=self.numerical_cols_,
                    index=df.index,
                )
            if self.scaler_ is not None:
                block = pd.DataFrame(
                    self.scaler_.transform(block),
                    columns=self.numerical_cols_,
                    index=df.index,
                )
            parts.append(block)

        if cfg.encode_categorical and self.categorical_cols_:
            cat_block = df[self.categorical_cols_].astype("object").fillna("__MISSING__")
            if cfg.encoding_method == "label":
                encoded: Dict[str, list] = {}
                for col in self.categorical_cols_:
                    le = self.encoders_[col]
                    known = set(str(c) for c in le.classes_)
                    # unseen labels -> -1 (documented behaviour)
                    encoded[col] = [
                        int(le.transform([v])[0]) if v in known else -1
                        for v in cat_block[col].astype(str)
                    ]
                parts.append(pd.DataFrame(encoded, index=df.index))
            else:
                if self.onehot_ is None:  # guard for mypy + runtime safety
                    raise RuntimeError("OneHotEncoder not fitted")
                arr = self.onehot_.transform(cat_block.astype(str))
                num = set(self.numerical_cols_)
                oh_names = [n for n in self.feature_names_ if n not in num]
                parts.append(pd.DataFrame(arr, columns=oh_names, index=df.index))

        out = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)
        out.columns = self.feature_names_

        if targets is not None:
            out = pd.concat([out, targets.loc[:, ~targets.columns.isin(out.columns)]], axis=1)
        return out

    def fit_transform(
        self, df: pd.DataFrame, targets: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Fit on ``df`` then transform it (convenience; use on train only)."""
        return self.fit(df).transform(df, targets=targets)
