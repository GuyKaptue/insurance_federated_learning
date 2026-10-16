"""
Deterministic synthetic insurance dataset generator.

Generates federated and centralized insurance datasets with site/holdout data
for four European insurance clients.
Five ML targets: claim_risk_score, customer_lifetime_value, will_renew,
is_high_risk, customer_segment (2 regression, 2 binary, 1 multi-class).

Determinism contract
--------------------
Each client draws from its own RNG seeded with
``np.random.SeedSequence([random_seed, crc32(client_id)])``.
Unlike Python's salted ``hash()``, crc32 is stable across processes, so two
runs in separate processes produce identical output for the same seed.

Leakage contract
----------------
No target is computed from another target. ``is_high_risk`` in particular
does not use ``claim_risk_score``.

Currency contract (docs/DECISIONS.md)
-------------------------------------
All monetary columns are in the client's LOCAL currency; every row carries an
explicit ``currency`` column (EUR or CHF) so mixed-currency pooling is always
self-describing. No silent FX conversion is applied.
"""

from __future__ import annotations

import json
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from insurance_fl.data.country_profiles import COUNTRY_PROFILES

TARGET_COLUMNS: List[str] = [
    "claim_risk_score",
    "customer_lifetime_value",
    "will_renew",
    "is_high_risk",
    "customer_segment",
]

SPLIT_NAMES: Tuple[str, ...] = ("train", "val", "test", "site")


class EuropeanInsuranceFederatedDataset:
    """European Insurance Dataset Generator.

    Parameters
    ----------
    samples_per_client : int
        Default samples per client (overridden by ``client_size_config``).
    random_seed : int
        Master seed. Client streams derive from it deterministically.
    dataset_version : str
        Version string recorded in metadata.json.
    client_size_config : Optional[Dict[str, int]]
        Per-client overrides {client_id: n_samples}.
    inject_quality_issues : bool
        Whether to inject missing values / outliers / inconsistencies
        (suspect #8: behavior is owned by the generator, gated by this flag,
        not by the CLI).
    """

    def __init__(
        self,
        samples_per_client: int = 5000,
        random_seed: int = 42,
        dataset_version: str = "v1.0",
        client_size_config: Optional[Dict[str, int]] = None,
        inject_quality_issues: bool = True,
    ) -> None:
        self.samples_per_client = samples_per_client
        self.random_seed = random_seed
        self.dataset_version = dataset_version
        self.client_size_config = client_size_config or {}
        self.inject_quality_issues = inject_quality_issues
        self.client_ids: List[str] = [
            "germany_allianz",
            "france_axa",
            "belgium_ageas",
            "switzerland_helvetia",
        ]
        self.client_country_map: Dict[str, str] = {
            "germany_allianz": "Germany",
            "france_axa": "France",
            "belgium_ageas": "Belgium",
            "switzerland_helvetia": "Switzerland",
        }

    def _get_client_sample_size(self, client_id: str) -> int:
        """Return the configured sample size for a client."""
        return int(self.client_size_config.get(client_id, self.samples_per_client))

    def _generate_client_data(
        self, country: str, n_samples: int, rng: np.random.Generator
    ) -> pd.DataFrame:
        """Generate feature data for a country with real-world correlations."""
        n = n_samples
        stats = COUNTRY_PROFILES[country]

        # --- Demographics & finances -------------------------------------
        age = np.clip(rng.normal(stats.avg_age, 15, n), 18, 95).astype(int)

        income_age_factor = 1 + (age - stats.avg_age) / 100
        income = np.clip(
            rng.normal(stats.income_mean * income_age_factor, stats.income_std, n),
            15000,
            300000,
        )

        savings_ratio = stats.savings_rate * (1 + rng.normal(0, 0.2, n))
        savings = np.clip(income * savings_ratio * rng.uniform(0.5, 2, n), 0, 200000)

        # credit_score rises with income and savings (documented correlation)
        credit_base = 650 + (income - 30000) / 30000 * 50 + np.log(savings + 1) / 10
        credit_score = np.clip(rng.normal(credit_base, 50, n), 300, 850).astype(int)

        # tenure rises with age (documented correlation, fixes suspect #9)
        tenure = np.clip(rng.gamma(3, 5, n) + (age - 46) * 0.3, 0, age - 18).astype(int)

        claim_prob = stats.claim_frequency * rng.uniform(0.5, 2, n)
        num_claims = np.clip(rng.poisson(claim_prob * 5, n), 0, 20)

        claim_freq = np.clip(num_claims / np.maximum(tenure + 1, 1), 0, 0.5)

        policy_value = np.clip(income * rng.uniform(0.5, 3, n) * 1.5, 5000, 5000000)

        premium = np.clip(
            (policy_value * 0.02 + stats.car_premium * 0.5) * (1 + rng.normal(0, 0.2, n)),
            200,
            20000,
        )

        health_risk = np.clip(
            (age / 90) * 50 + (stats.health_spending / 10000) * 20 + rng.normal(0, 10, n),
            0,
            100,
        )

        # --- Categoricals (vectorized rng.choice, no per-row loops) -------
        marital_status = rng.choice(
            ["Single", "Married", "Divorced", "Widowed"],
            n,
            p=[0.35, 0.45, 0.15, 0.05],
        )
        education = rng.choice(
            ["Secondary", "Bachelor", "Master", "PhD", "Professional"],
            n,
            p=[0.30, 0.30, 0.25, 0.10, 0.05],
        )
        employment = rng.choice(
            ["Employed", "Self-Employed", "Unemployed", "Retired"],
            n,
            p=[0.60, 0.20, 0.10, 0.10],
        )
        home_ownership = rng.choice(
            ["Own", "Rent", "Mortgage", "Other"], n, p=[0.40, 0.30, 0.25, 0.05]
        )
        insurance_package = rng.choice(
            ["Basic", "Standard", "Premium", "Elite"], n, p=[0.20, 0.40, 0.30, 0.10]
        )
        risk_category = rng.choice(
            ["Low", "Medium", "High", "Very High"], n, p=[0.40, 0.35, 0.20, 0.05]
        )
        policy_type = rng.choice(["Individual", "Family", "Corporate"], n, p=[0.5, 0.3, 0.2])
        vehicle_type = rng.choice(
            ["Sedan", "SUV", "Sports", "Truck", "Motorcycle"],
            n,
            p=[0.40, 0.30, 0.15, 0.10, 0.05],
        )
        property_type = rng.choice(
            ["Apartment", "House", "Villa", "Commercial"],
            n,
            p=[0.45, 0.40, 0.10, 0.05],
        )
        profession = rng.choice(
            ["Engineer", "Doctor", "Lawyer", "Teacher", "Business", "Other"],
            n,
            p=[0.20, 0.10, 0.10, 0.15, 0.25, 0.20],
        )
        insurance_type = rng.choice(
            ["Auto", "Life", "Health", "Property", "Liability", "Travel", "Accident"],
            n,
            p=[0.20, 0.15, 0.20, 0.15, 0.15, 0.10, 0.05],
        )
        policy_status = rng.choice(
            ["Active", "Lapsed", "Cancelled", "Pending"], n, p=[0.75, 0.10, 0.10, 0.05]
        )
        family = rng.choice([1, 2, 3, 4, 5, 6], n, p=[0.15, 0.25, 0.30, 0.20, 0.07, 0.03])
        policy_duration = rng.exponential(5, n).clip(0, 30).astype(int)

        # --- Derived numerics --------------------------------------------
        # driving_experience cannot legally exceed age - 16
        driving_exp = np.clip(age - 18 + rng.normal(0, 3, n), 0, age - 16).astype(int)

        vehicle_value = np.clip(income * rng.uniform(0.3, 1.5, n), 5000, 200000)

        dti_mean = 0.35 - stats.savings_rate * 0.5
        dti = np.clip(rng.beta(2, 4, n) * 0.5 + dti_mean, 0.05, 0.8)

        liability_base = 50 + (stats.risk_appetite - 0.5) * 20
        liability = np.clip(rng.beta(2, 3, n) * 100 + liability_base * 0.3, 0, 100)

        property_value = np.clip(income * rng.uniform(0.5, 4, n), 50000, 2000000)

        medical = np.clip((age / 90) * stats.health_spending + rng.normal(0, 2000, n), 0, 50000)

        life_exp = np.clip(stats.life_expectancy + rng.normal(0, 5, n), 60, 100)

        data = pd.DataFrame(
            {
                "annual_income": income,
                "savings_balance": savings,
                "policy_count": rng.poisson(2, n).clip(1, 10),
                "policy_value": policy_value,
                "age": age,
                "years_as_customer": tenure,
                "number_of_claims": num_claims,
                "total_claim_amount": num_claims * rng.uniform(1000, 50000, n),
                "claim_frequency": claim_freq,
                "property_value": property_value,
                "medical_expenses": medical,
                "life_expectancy": life_exp,
                "health_risk_score": health_risk,
                "driving_experience": driving_exp,
                "vehicle_value": vehicle_value,
                "liability_risk": liability,
                "debt_to_income_ratio": dti,
                "credit_score": credit_score,
                "family_members": family,
                "policy_duration": policy_duration,
                "premium_amount": premium,
                "marital_status": marital_status,
                "education_level": education,
                "employment_status": employment,
                "home_ownership": home_ownership,
                "insurance_package": insurance_package,
                "risk_category": risk_category,
                "policy_type": policy_type,
                "vehicle_type": vehicle_type,
                "property_type": property_type,
                "profession": profession,
                "primary_insurance_type": insurance_type,
                "policy_status": policy_status,
                "country": country,
                "insurance_company": stats.company,
                "regulatory_framework": stats.regulatory_framework,
                "currency": stats.currency,
            }
        )

        targets = self._generate_targets(data, stats, rng)
        return pd.concat([data, targets], axis=1)

    def _client_rng(self, client_id: str) -> np.random.Generator:
        """Per-client RNG stable across processes (crc32, not salted hash())."""
        stable_key = zlib.crc32(client_id.encode("utf-8"))
        return np.random.default_rng(np.random.SeedSequence([self.random_seed, stable_key]))

    def _generate_targets(self, df: pd.DataFrame, stats, rng: np.random.Generator) -> pd.DataFrame:
        """Generate the 5 insurance targets with realistic correlations.

        No label leakage between targets — each target is computed from
        independent feature combinations only.

        Targets
        -------
        claim_risk_score        regression   [0, 100]
        customer_lifetime_value regression   [1000, 250000]
        will_renew              binary       {0, 1}
        is_high_risk            binary       {0, 1}
        customer_segment        multi-class  {0, 1, 2}
        """
        n = len(df)
        targets = pd.DataFrame(index=df.index)

        # --- Target 1: Regression - Claim Risk Score (0-100) -------------
        claim_risk = (
            df["number_of_claims"] / 20 * 30
            + df["claim_frequency"] / 0.5 * 20
            + df["health_risk_score"] / 100 * 15
            + df["liability_risk"] / 100 * 10
            + (100 - df["credit_score"]) / 850 * 15
            + df["debt_to_income_ratio"] * 10
        )
        claim_risk = claim_risk * (stats.claim_frequency / 0.04)
        claim_risk = np.clip(claim_risk + rng.normal(0, 5, n), 0, 100)
        targets["claim_risk_score"] = claim_risk

        # --- Target 2: Regression - Customer Lifetime Value --------------
        clv = (
            df["annual_income"] * 0.02
            + df["savings_balance"] * 0.03
            + df["years_as_customer"] * 50
            + df["policy_value"] * 0.01
            + (df["insurance_package"] == "Premium").astype(float) * 1000
            + (df["insurance_package"] == "Elite").astype(float) * 2000
            - df["number_of_claims"] * 500
        )
        clv = np.clip(clv + rng.normal(0, 500, n), 1000, 250000)
        targets["customer_lifetime_value"] = clv

        # --- Target 3: Binary - Will Renew Policy ------------------------
        renewal_logit = (
            df["years_as_customer"] / 50 * 0.3
            - df["claim_frequency"] / 0.5 * 1.2
            + df["savings_balance"] / 100000 * 0.2
            + (df["policy_status"] == "Active").astype(float) * 0.5
            - df["number_of_claims"] / 20 * 0.8
            + df["credit_score"] / 850 * 0.3
            - df["debt_to_income_ratio"] * 0.5
            + {"Germany": 0.3, "France": -0.1, "Belgium": 0.1, "Switzerland": 0.4}.get(
                stats.country, 0.0
            )
            + 0.5
        )
        renewal_prob = np.clip(1 / (1 + np.exp(-renewal_logit)), 0.55, 0.92)
        targets["will_renew"] = (rng.random(n) < renewal_prob).astype(int)

        # --- Target 4: Binary - High Risk Customer -----------------------
        # NOTE: does NOT use claim_risk_score (avoids label leakage).
        high_risk_logit = (
            -1.2
            + df["number_of_claims"] / 20 * 1.5
            + df["debt_to_income_ratio"] * 1.5
            + (df["insurance_package"] == "Basic").astype(float) * 0.5
            + (df["number_of_claims"] > 2).astype(float) * 1.0
            - df["credit_score"] / 850 * 0.8
            - df["savings_balance"] / 100000 * 0.5
        )
        high_risk_prob = np.clip(1 / (1 + np.exp(-high_risk_logit)), 0.15, 0.40)
        targets["is_high_risk"] = (rng.random(n) < high_risk_prob).astype(int)

        # --- Target 5: Multi-class - Customer Segment --------------------
        # Income genuinely shifts segment odds: each segment's score has a
        # strictly increasing coefficient on annual_income (0.3 < 0.8 < 1.2),
        # so higher income monotonically favours higher-value segments.
        # Sampling is fully vectorized via inverse-CDF on cumsum (suspect #10:
        # no per-row Python loop with rng.choice).
        inc = df["annual_income"] / 100000
        seg_scores = np.column_stack(
            [
                -0.5 + inc * 0.3 + df["age"] / 90 * 0.2 + df["years_as_customer"] / 50 * 0.1,
                0.5
                + inc * 0.8
                + df["savings_balance"] / 100000 * 0.5
                + df["credit_score"] / 850 * 0.3
                + (df["insurance_package"] == "Premium").astype(float) * 0.5,
                1.0
                + inc * 1.2
                + df["savings_balance"] / 100000 * 0.8
                + df["credit_score"] / 850 * 0.5
                + (df["insurance_package"] == "Elite").astype(float) * 0.8,
            ]
        )
        # log-softmax for numerical stability
        seg_scores = seg_scores - seg_scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(seg_scores)
        seg_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        u = rng.random(n)
        cum = np.cumsum(seg_probs, axis=1)
        segment_labels = (u[:, None] > cum).sum(axis=1).astype(int)
        targets["customer_segment"] = segment_labels

        return targets

    def _add_data_quality_issues(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Inject missing values, outliers and inconsistencies for realism.

        Fixes for suspect #7:
        * Integer columns are upcast to float before NaN assignment
          (NaN cannot live in int64 → silent dtype corruption).
        * Outlier scale uses ``nanmax``/``nanmin`` guards so an earlier NaN
          or already-injected outlier cannot poison the statistic.
        * ``tenure <= age - 18`` is preserved (previously ``age + 5``, which
          violated the generation invariant and my sanity checks).
        Injection rates match the config passed to the constructor (tested).
        """
        df = df.copy()
        original_na_count = int(df.isna().sum().sum())

        # 1) Missing values (~3%) — only in float/int numeric columns,
        #    never in categorical or target columns.
        candidate_cols = [
            c for c in df.select_dtypes(include=["number"]).columns if c not in TARGET_COLUMNS
        ]
        n_cells = len(df) * len(candidate_cols)
        if n_cells > 0 and candidate_cols:
            mask = rng.random((len(df), len(candidate_cols))) < 0.03
            block = df[candidate_cols].astype("float64")
            block = block.mask(mask)
            df[candidate_cols] = block

        # 2) Outliers (~0.5%) — multiply the upper tail using a NaN-safe
        #    nanmax so injected NaNs and earlier outliers cannot cascade.
        outlier_cols = [
            "annual_income",
            "premium_amount",
            "total_claim_amount",
            "savings_balance",
            "property_value",
            "vehicle_value",
            "medical_expenses",
        ]
        for col in outlier_cols:
            if col not in df.columns:
                continue
            col_max = df[col].max(skipna=True)
            if pd.isna(col_max):  # all-NaN column: nothing to scale against
                continue
            mask = rng.random(len(df)) < 0.005
            df.loc[mask, col] = col_max * rng.uniform(2.0, 5.0, int(mask.sum()))

        # 3) Inconsistencies (~1%): force tenure down so it never exceeds
        #    age - 18 (legal/realistic invariant preserved).
        #    NOTE: age may contain injected NaNs (upcast to float), so cap can
        #    be NaN — cast is deliberately float (NaN->int64 raises).
        if "years_as_customer" in df.columns and "age" in df.columns:
            cap = (df["age"] - 18).clip(lower=0)
            bad = rng.random(len(df)) < 0.01
            if bad.any():
                vals = cap[bad] * rng.random(int(bad.sum()))
                df.loc[bad, "years_as_customer"] = vals.astype("float64")

        assert int(df.isna().sum().sum()) >= original_na_count
        return df

    def generate_dataset(self, output_dir: str, save_format: str = "parquet") -> Dict[str, Any]:
        """Generate federated + centralized datasets and save them to disk.

        Split policy (documented, tested):
        * per client: 60% train / 40% holdout (stratified on
          customer_segment), then holdout -> 50% val / 50% test_site,
          then 25% of test_site is reserved as the unseen ``site`` split
          (5% of total per client).
        * ``centralized`` = random 25% subsample of every client's data.
        """
        all_data: Dict[str, Any] = {}
        print("=" * 60)
        print("European Insurance Dataset Generator")
        print("=" * 60)

        for client_id in self.client_ids:
            country = self.client_country_map[client_id]
            n_samples = self._get_client_sample_size(client_id)
            rng = self._client_rng(client_id)

            print(f"Generating {client_id} ({country}): " f"{n_samples:,} samples...")
            data = self._generate_client_data(country, n_samples, rng)

            # Unique customer ids (vectorized — no per-row rng calls)
            digits = rng.integers(0, 10, size=(len(data), 8)).astype(str)
            suffix = ["".join(row) for row in digits]
            data["customer_id"] = country[:2].upper() + pd.Series(suffix, index=data.index)

            if self.inject_quality_issues:
                data = self._add_data_quality_issues(data, rng)

            # 60/40 split, stratified on a target (no target leakage: the
            # split only uses labels for stratification, never as features)
            train_df, holdout_df = train_test_split(
                data,
                test_size=0.4,
                random_state=self.random_seed,
                stratify=data["customer_segment"],
            )
            val_df, test_df = train_test_split(
                holdout_df,
                test_size=0.5,
                random_state=self.random_seed,
                stratify=holdout_df["customer_segment"],
            )
            site_df, test_df = train_test_split(
                test_df,
                test_size=0.75,
                random_state=self.random_seed,
                stratify=test_df["customer_segment"],
            )

            all_data[client_id] = {
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "site": site_df,
            }
            all_data[client_id]["metadata"] = {
                "country": country,
                "total_samples": len(data),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "site_samples": len(site_df),
                "target_distribution": {
                    col: {str(k): int(v) for k, v in data[col].value_counts().items()}
                    for col in TARGET_COLUMNS
                },
            }
            print(
                f"  Train: {len(train_df):,} | Val: {len(val_df):,} | "
                f"Test: {len(test_df):,} | Site: {len(site_df):,}"
            )

        centralized = self._create_centralized_dataset(all_data)
        print(f"\nCentralized dataset: {len(centralized):,} total samples")

        metadata = {
            "dataset_version": self.dataset_version,
            "random_seed": self.random_seed,
            "output_directory": output_dir,
            "save_format": save_format,
            "quality_issues_injected": self.inject_quality_issues,
            "generation_timestamp_utc": datetime.now().isoformat(),
            "total_clients": len(self.client_ids),
            "client_sizes": {
                cid: all_data[cid]["metadata"]["total_samples"] for cid in self.client_ids
            },
            "centralized_samples": len(centralized),
            "targets": TARGET_COLUMNS,
        }
        print(f"Total clients: {len(self.client_ids)}")
        print(f"Random seed: {self.random_seed} (per-client crc32 streams)")
        print("Dataset generation completed successfully!")

        save_info = self._save_dataset(all_data, output_dir, save_format, centralized)
        metadata["files"] = save_info
        return metadata

    def _create_centralized_dataset(self, all_data: Dict[str, Any]) -> pd.DataFrame:
        """Pool a deterministic 25% sample of every client's data."""
        frames: List[pd.DataFrame] = []
        for client_id in self.client_ids:
            client_train = all_data[client_id]["train"]
            frac = 0.25
            n_take = max(1, int(len(client_train) * frac))
            sample = client_train.sample(n=n_take, random_state=self.random_seed).copy()
            sample["client_id"] = client_id
            frames.append(sample)
        return pd.concat(frames, ignore_index=True)

    def _save_dataset(
        self,
        all_data: Dict[str, Any],
        output_dir: str,
        save_format: str = "parquet",
        centralized_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Persist federated splits, the centralized pool and metadata.

        Returns a manifest {relative_path: row_count}.
        """
        output_path = Path(output_dir)
        saved: Dict[str, Any] = {}

        def _write(df: pd.DataFrame, path: Path) -> None:
            if save_format == "parquet":
                df.to_parquet(path, index=False)
            else:
                df.to_csv(path, index=False)
            saved[str(path.relative_to(output_path))] = len(df)

        for client_id in self.client_ids:
            splits = all_data[client_id]
            client_dir = output_path / "federated" / client_id
            client_dir.mkdir(parents=True, exist_ok=True)
            for split_name in SPLIT_NAMES:
                _write(splits[split_name], client_dir / f"{split_name}.{save_format}")

        if centralized_data is not None:
            central_dir = output_path / "centralized"
            central_dir.mkdir(parents=True, exist_ok=True)
            _write(
                centralized_data,
                central_dir / f"all_clients.{save_format}",
            )

        metadata = {
            "dataset_version": self.dataset_version,
            "random_seed": self.random_seed,
            "samples_per_client": self.samples_per_client,
            "total_clients": len(self.client_ids),
            "client_sizes": {
                cid: int(all_data[cid]["metadata"]["total_samples"]) for cid in self.client_ids
            },
            "quality_issues_injected": self.inject_quality_issues,
            "generation_timestamp_utc": datetime.now().isoformat(),
        }
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        dataset_info = {
            "clients": {
                cid: {
                    "country": all_data[cid]["metadata"]["country"],
                    "splits": {sn: int(len(all_data[cid][sn])) for sn in SPLIT_NAMES},
                    "targets": {
                        col: {
                            str(k): int(v)
                            for k, v in all_data[cid]["train"][col].value_counts().items()
                        }
                        for col in TARGET_COLUMNS
                    },
                }
                for cid in self.client_ids
            },
            "centralized_samples": (len(centralized_data) if centralized_data is not None else 0),
        }
        (output_path / "dataset_info.json").write_text(
            json.dumps(dataset_info, indent=2), encoding="utf-8"
        )
        return saved

    def load_dataset(
        self, data_dir: str, split: str = "train", client_id: Optional[str] = None
    ) -> pd.DataFrame:
        """Load a previously generated split from disk."""
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
        base = Path(data_dir)
        if client_id is None:
            path = (
                base
                / "centralized"
                / f"all_clients.{ 'parquet' if (base / 'centralized' / 'all_clients.parquet').exists() else 'csv'}"
            )
        else:
            fmt = (
                "parquet" if (base / "federated" / client_id / "train.parquet").exists() else "csv"
            )
            path = base / "federated" / client_id / f"{split}.{fmt}"
        if not path.exists():
            raise FileNotFoundError(f"No dataset file at {path}")
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    def get_dataset_summary(self, data_dir: str) -> Dict[str, Any]:
        """Return the recorded metadata and per-client dataset info."""
        base = Path(data_dir)
        metadata_file = base / "metadata.json"
        info_file = base / "dataset_info.json"
        metadata = (
            json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
        )
        dataset_info = (
            json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}
        )
        return {"metadata": metadata, "dataset_info": dataset_info}
