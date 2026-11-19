"""
European Insurance Federated Dataset Generator
LEGACY (deprecated): saves to a non-canonical synthetic path by default; do not use for canonical data.
Generates both federated (client-specific) and centralized datasets
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

_LEGACY_DEPRECATION = (
    "src.synthetic.data_generator is a legacy/non-canonical generator "
    "(claim_risk-coupled/older behavior) and is deprecated. "
    "Use the official CLI `insurance-fl-generate` backed by "
    "`insurance_fl.data.generator.EuropeanInsuranceFederatedDataset`."
)


class EuropeanInsuranceFederatedDataset:
    """
    Professional European Insurance Dataset Generator
    Based on real-world statistics for Germany, France, Belgium, Switzerland
    Generates both federated (client-specific) and centralized datasets
    """

    def __init__(self, samples_per_client=5000, random_seed=42, dataset_version="v1.0"):
        warnings.warn(_LEGACY_DEPRECATION, DeprecationWarning, stacklevel=2)
        self.samples_per_client = samples_per_client
        self.random_seed = random_seed
        self.dataset_version = dataset_version
        np.random.seed(random_seed)

        # Real-world country statistics
        self.country_stats = {
            'Germany': {
                'company': 'Allianz',
                'income_mean': 50000, 'income_std': 15000,
                'savings_rate': 0.107, 'savings_mean': 3568,
                'claim_frequency': 0.044,  # 4.4% auto claims
                'life_expectancy': 81,
                'avg_age': 46.5,
                'population': 83200000,
                'gdp_per_capita': 48000,
                'car_premium': 800,
                'health_spending': 6000,
                'risk_appetite': 0.7,
                'digital_adoption': 0.85,
                'market_share': 0.25
            },
            'France': {
                'company': 'AXA',
                'income_mean': 45000, 'income_std': 14000,
                'savings_rate': 0.184, 'savings_mean': 2876,
                'claim_frequency': 0.055,  # Higher claim incidence
                'life_expectancy': 82,
                'avg_age': 42.3,
                'population': 67750000,
                'gdp_per_capita': 43000,
                'car_premium': 900,
                'health_spending': 5500,
                'risk_appetite': 0.8,
                'digital_adoption': 0.80,
                'market_share': 0.22
            },
            'Belgium': {
                'company': 'Ageas',
                'income_mean': 42000, 'income_std': 13000,
                'savings_rate': 0.129, 'savings_mean': 2778,
                'claim_frequency': 0.038,  # Lower claims
                'life_expectancy': 81,
                'avg_age': 41.8,
                'population': 11590000,
                'gdp_per_capita': 41000,
                'car_premium': 750,
                'health_spending': 5000,
                'risk_appetite': 0.6,
                'digital_adoption': 0.82,
                'market_share': 0.18
            },
            'Switzerland': {
                'company': 'Helvetia',
                'income_mean': 65000, 'income_std': 20000,
                'savings_rate': 0.181, 'savings_mean': 5908,
                'claim_frequency': 0.030,  # Lowest claims
                'life_expectancy': 84,
                'avg_age': 43.2,
                'population': 8700000,
                'gdp_per_capita': 65000,
                'car_premium': 1200,
                'health_spending': 7000,
                'risk_appetite': 0.5,
                'digital_adoption': 0.90,
                'market_share': 0.20
            }
        }

        # Feature definitions
        self.feature_definitions = {
            'numerical': [
                'annual_income', 'savings_balance', 'policy_count',
                'policy_value', 'age', 'years_as_customer',
                'number_of_claims', 'total_claim_amount', 'claim_frequency',
                'property_value', 'medical_expenses', 'life_expectancy',
                'health_risk_score', 'driving_experience', 'vehicle_value',
                'liability_risk', 'debt_to_income_ratio', 'credit_score',
                'family_members', 'policy_duration', 'premium_amount'
            ],
            'categorical': [
                'marital_status', 'education_level', 'employment_status',
                'home_ownership', 'insurance_package', 'risk_category',
                'policy_type', 'vehicle_type', 'property_type', 'profession',
                'primary_insurance_type', 'policy_status'
            ]
        }

    def _generate_client_data(self, country):
        """Generate data for a specific country with real-world correlations"""

        n = self.samples_per_client
        stats = self.country_stats[country]

        # --- Generate Correlated Features ---

        # 1. Age (based on country demographics)
        age = np.random.normal(stats['avg_age'], 15, n)
        age = np.clip(age, 18, 95).astype(int)

        # 2. Annual Income (correlated with age and country GDP)
        income_age_factor = 1 + (age - stats['avg_age']) / 100
        income = np.random.normal(
            stats['income_mean'] * income_age_factor,
            stats['income_std'],
            n
        )
        income = np.clip(income, 15000, 300000)

        # 3. Savings Balance (correlated with income and country savings rate)
        savings_ratio = stats['savings_rate'] * (1 + np.random.normal(0, 0.2, n))
        savings_base = income * savings_ratio * np.random.uniform(0.5, 2, n)
        savings = np.random.normal(stats['savings_mean'], stats['savings_mean'] * 0.4, n)
        savings = np.where(savings > 0, savings, 0)
        savings = np.clip(savings, 0, 200000)

        # 4. Credit Score (correlated with income and savings)
        credit_base = 650 + (income - 30000) / 30000 * 50 + np.log(savings + 1) / 10
        credit_score = np.random.normal(credit_base, 50, n)
        credit_score = np.clip(credit_score, 300, 850).astype(int)

        # 5. Years as Customer (correlated with age)
        years_customer = np.random.gamma(3, 5, n)
        years_customer = np.clip(years_customer, 0, age - 18).astype(int)

        # 6. Number of Claims (based on country claim frequency)
        claim_prob = stats['claim_frequency'] * np.random.uniform(0.5, 2, n)
        num_claims = np.random.poisson(claim_prob * 5, n)
        num_claims = np.clip(num_claims, 0, 20)

        # 7. Claim Frequency (claims per year)
        claim_freq = num_claims / np.maximum(years_customer + 1, 1)
        claim_freq = np.clip(claim_freq, 0, 0.5)

        # 8. Policy Value (correlated with income and claims)
        policy_value = income * np.random.uniform(0.5, 3, n) * 1.5
        policy_value = np.clip(policy_value, 5000, 5000000)

        # 9. Premium Amount (based on policy value and country)
        premium_base = policy_value * 0.02 + stats['car_premium'] * 0.5
        premium = premium_base * (1 + np.random.normal(0, 0.2, n))
        premium = np.clip(premium, 200, 20000)

        # 10. Health Risk Score (correlated with age and health spending)
        health_risk = (age / 90) * 50 + (stats['health_spending'] / 10000) * 20
        health_risk += np.random.normal(0, 10, n)
        health_risk = np.clip(health_risk, 0, 100)

        # 11. Driving Experience (correlated with age)
        driving_exp = age - 18 + np.random.normal(0, 3, n)
        driving_exp = np.clip(driving_exp, 0, 60).astype(int)

        # 12. Vehicle Value (correlated with income)
        vehicle_value = income * np.random.uniform(0.3, 1.5, n)
        vehicle_value = np.clip(vehicle_value, 5000, 200000)

        # 13. Debt-to-Income Ratio (inverse correlation with savings rate)
        dti_mean = 0.35 - stats['savings_rate'] * 0.5
        dti = np.random.beta(2, 4, n) * 0.5 + dti_mean
        dti = np.clip(dti, 0.05, 0.8)

        # 14. Liability Risk (higher in litigious countries)
        liability_base = 50 + (stats['risk_appetite'] - 0.5) * 20
        liability = np.random.beta(2, 3, n) * 100 + liability_base * 0.3
        liability = np.clip(liability, 0, 100)

        # 15. Property Value (correlated with income)
        property_value = income * np.random.uniform(2, 10, n)
        property_value = np.clip(property_value, 100000, 5000000)

        # 16. Medical Expenses (based on health spending)
        medical = np.random.gamma(2, stats['health_spending'] / 3, n)
        medical = np.clip(medical, 0, 150000)

        # 17. Life Expectancy (country-specific)
        life_exp = stats['life_expectancy'] + np.random.normal(0, 5, n)
        life_exp = np.clip(life_exp, 60, 95)

        # 18. Policy Duration (years)
        policy_duration = np.random.gamma(2, 3, n)
        policy_duration = np.clip(policy_duration, 1, 30).astype(int)

        # 19. Policy Count (correlated with income)
        policy_count = np.random.poisson(2 + income / 50000, n)
        policy_count = np.clip(policy_count, 1, 12)

        # 20. Family Members
        family = np.random.poisson(1.5, n)
        family = np.clip(family, 1, 8)

        # --- Generate Categorical Features with Country Biases ---

        # Marital Status
        marital_probs = {
            'Germany': [0.25, 0.50, 0.15, 0.10],
            'France': [0.30, 0.45, 0.15, 0.10],
            'Belgium': [0.28, 0.48, 0.14, 0.10],
            'Switzerland': [0.22, 0.52, 0.16, 0.10]
        }
        marital_status = np.random.choice(
            ['Single', 'Married', 'Divorced', 'Widowed'],
            n,
            p=marital_probs[country]
        )

        # Education Level
        edu_probs = {
            'Germany': [0.15, 0.35, 0.30, 0.15, 0.05],
            'France': [0.20, 0.30, 0.25, 0.20, 0.05],
            'Belgium': [0.18, 0.32, 0.28, 0.17, 0.05],
            'Switzerland': [0.12, 0.30, 0.35, 0.18, 0.05]
        }
        education = np.random.choice(
            ['Secondary', 'Bachelor', 'Master', 'PhD', 'Professional'],
            n,
            p=edu_probs[country]
        )

        # Employment Status
        emp_probs = {
            'Germany': [0.65, 0.15, 0.05, 0.15],
            'France': [0.60, 0.18, 0.07, 0.15],
            'Belgium': [0.62, 0.16, 0.06, 0.16],
            'Switzerland': [0.68, 0.14, 0.04, 0.14]
        }
        employment = np.random.choice(
            ['Employed', 'Self-Employed', 'Unemployed', 'Retired'],
            n,
            p=emp_probs[country]
        )

        # Home Ownership
        home_probs = {
            'Germany': [0.35, 0.30, 0.30, 0.05],
            'France': [0.40, 0.25, 0.30, 0.05],
            'Belgium': [0.38, 0.28, 0.30, 0.04],
            'Switzerland': [0.30, 0.35, 0.30, 0.05]
        }
        home_ownership = np.random.choice(
            ['Own', 'Rent', 'Mortgage', 'Other'],
            n,
            p=home_probs[country]
        )

        # Insurance Package
        package_probs = {
            'Germany': [0.25, 0.35, 0.30, 0.10],
            'France': [0.20, 0.30, 0.35, 0.15],
            'Belgium': [0.30, 0.35, 0.25, 0.10],
            'Switzerland': [0.15, 0.30, 0.40, 0.15]
        }
        insurance_package = np.random.choice(
            ['Basic', 'Standard', 'Premium', 'Elite'],
            n,
            p=package_probs[country]
        )

        # Risk Category (correlated with claim_freq and health_risk)
        risk_score = claim_freq * 2 + health_risk / 100 + liability / 100
        risk_categories = ['Low', 'Medium', 'High', 'Very High']
        risk_probs = [0.3, 0.35, 0.25, 0.10] if stats['risk_appetite'] < 0.6 else [0.2, 0.3, 0.3, 0.2]
        risk_category = np.random.choice(risk_categories, n, p=risk_probs)

        # Vehicle Type (country preferences)
        vehicle_probs = {
            'Germany': [0.30, 0.35, 0.15, 0.10, 0.10],
            'France': [0.35, 0.30, 0.12, 0.08, 0.15],
            'Belgium': [0.32, 0.33, 0.13, 0.09, 0.13],
            'Switzerland': [0.28, 0.38, 0.15, 0.10, 0.09]
        }
        vehicle_type = np.random.choice(
            ['Sedan', 'SUV', 'Sports', 'Truck', 'Motorcycle'],
            n,
            p=vehicle_probs[country]
        )

        # Property Type
        property_probs = {
            'Germany': [0.35, 0.40, 0.15, 0.10],
            'France': [0.40, 0.35, 0.15, 0.10],
            'Belgium': [0.38, 0.38, 0.14, 0.10],
            'Switzerland': [0.30, 0.45, 0.15, 0.10]
        }
        property_type = np.random.choice(
            ['Apartment', 'House', 'Villa', 'Commercial'],
            n,
            p=property_probs[country]
        )

        # Profession
        profession_probs = {
            'Germany': [0.20, 0.15, 0.12, 0.10, 0.25, 0.18],
            'France': [0.18, 0.16, 0.14, 0.12, 0.22, 0.18],
            'Belgium': [0.19, 0.15, 0.13, 0.11, 0.24, 0.18],
            'Switzerland': [0.22, 0.18, 0.15, 0.10, 0.20, 0.15]
        }
        profession = np.random.choice(
            ['Engineer', 'Doctor', 'Lawyer', 'Teacher', 'Business', 'Other'],
            n,
            p=profession_probs[country]
        )

        # Primary Insurance Type (product mix)
        product_probs = {
            'Germany': [0.20, 0.15, 0.20, 0.15, 0.15, 0.10, 0.05],
            'France': [0.18, 0.18, 0.22, 0.12, 0.12, 0.12, 0.06],
            'Belgium': [0.22, 0.12, 0.18, 0.18, 0.14, 0.10, 0.06],
            'Switzerland': [0.15, 0.20, 0.15, 0.20, 0.15, 0.10, 0.05]
        }
        insurance_type = np.random.choice(
            ['Auto', 'Life', 'Health', 'Property', 'Liability', 'Travel', 'Accident'],
            n,
            p=product_probs[country]
        )

        # Policy Status
        policy_status = np.random.choice(
            ['Active', 'Lapsed', 'Cancelled', 'Pending'],
            n,
            p=[0.75, 0.10, 0.10, 0.05]
        )

        # --- Combine into DataFrame ---

        data = pd.DataFrame({
            # Numerical features
            'annual_income': income,
            'savings_balance': savings,
            'policy_count': policy_count,
            'policy_value': policy_value,
            'age': age,
            'years_as_customer': years_customer,
            'number_of_claims': num_claims,
            'total_claim_amount': num_claims * np.random.uniform(1000, 50000, n),
            'claim_frequency': claim_freq,
            'property_value': property_value,
            'medical_expenses': medical,
            'life_expectancy': life_exp,
            'health_risk_score': health_risk,
            'driving_experience': driving_exp,
            'vehicle_value': vehicle_value,
            'liability_risk': liability,
            'debt_to_income_ratio': dti,
            'credit_score': credit_score,
            'family_members': family,
            'policy_duration': policy_duration,
            'premium_amount': premium,

            # Categorical features
            'marital_status': marital_status,
            'education_level': education,
            'employment_status': employment,
            'home_ownership': home_ownership,
            'insurance_package': insurance_package,
            'risk_category': risk_category,
            'policy_type': np.random.choice(['Individual', 'Family', 'Corporate'], n, p=[0.5, 0.3, 0.2]),
            'vehicle_type': vehicle_type,
            'property_type': property_type,
            'profession': profession,
            'primary_insurance_type': insurance_type,
            'policy_status': policy_status,

            # Metadata
            'country': country,
            'insurance_company': stats['company'],
            'regulatory_framework': ['BaFin' if country=='Germany' else
                                    'ACPR' if country=='France' else
                                    'NBB/FSMA' if country=='Belgium' else 'FINMA'] * n,
            'currency': ['€'] * n if country != 'Switzerland' else ['CHF'] * n
        })

        # Generate correlated targets
        targets = self._generate_targets(data, country)
        data = pd.concat([data, targets], axis=1)

        return data

    def _generate_targets(self, df, country):
        """Generate 5 insurance targets with real correlations"""

        n = len(df)
        stats = self.country_stats[country]
        targets = pd.DataFrame()

        # --- Target 1: Regression - Claim Risk Score (0-100) ---
        # Real correlation with claims, age, income, and risk factors
        claim_risk = (
            df['number_of_claims'] / 20 * 30 +
            df['claim_frequency'] / 0.5 * 20 +
            df['health_risk_score'] / 100 * 15 +
            df['liability_risk'] / 100 * 10 +
            (100 - df['credit_score']) / 850 * 15 +
            df['debt_to_income_ratio'] * 10
        )
        # Country adjustment
        claim_risk *= (stats['claim_frequency'] / 0.04)
        claim_risk = np.clip(claim_risk + np.random.normal(0, 5, n), 0, 100)
        targets['claim_risk_score'] = claim_risk

        # --- Target 2: Regression - Customer Lifetime Value (CLV) ---
        # Real correlation with income, savings, loyalty, and policy value
        clv = (
            df['annual_income'] * 0.02 +
            df['savings_balance'] * 0.03 +
            df['years_as_customer'] * 50 +
            df['policy_value'] * 0.01 +
            (df['insurance_package'] == 'Premium').astype(float) * 1000 +
            (df['insurance_package'] == 'Elite').astype(float) * 2000 -
            df['number_of_claims'] * 500
        )
        clv = np.clip(clv + np.random.normal(0, 500, n), 1000, 250000)
        targets['customer_lifetime_value'] = clv

        # --- Target 3: Binary Classification - Will Renew Policy ---
        # Real correlation with customer satisfaction proxies
        renewal_prob = 1 / (1 + np.exp(-(
            1.5 +
            df['years_as_customer'] / 50 * 1.2 -
            df['claim_frequency'] / 0.5 * 1.5 +
            (df['policy_status'] == 'Active').astype(float) * 1.0 +
            (df['insurance_package'] == 'Elite').astype(float) * 0.5 -
            df['number_of_claims'] / 20 * 1.0 +
            df['savings_balance'] / 100000 * 0.3
        )))

        # Country-specific renewal behavior
        renewal_bias = {'Germany': 0.1, 'France': -0.05, 'Belgium': 0.05, 'Switzerland': 0.15}
        renewal_prob += renewal_bias[country]
        renewal_prob = np.clip(renewal_prob, 0.05, 0.95)
        targets['will_renew'] = (np.random.random(n) < renewal_prob).astype(int)
        targets['renewal_probability'] = renewal_prob

        # --- Target 4: Binary Classification - High Risk Customer ---
        # FIXED: Use claim_risk variable directly instead of df['claim_risk_score']
        high_risk_prob = 1 / (1 + np.exp(-(
            -1.2 +
            claim_risk / 100 * 2.0 +
            df['debt_to_income_ratio'] * 1.5 +
            (df['insurance_package'] == 'Basic').astype(float) * 0.5 +
            (df['number_of_claims'] > 2).astype(float) * 1.0 -
            df['credit_score'] / 850 * 0.5 -
            df['savings_balance'] / 100000 * 0.3
        )))

        # Ensure reasonable balance (40-50% positive)
        high_risk_prob = np.clip(high_risk_prob, 0.2, 0.7)
        targets['is_high_risk'] = (np.random.random(n) < high_risk_prob).astype(int)
        targets['risk_probability'] = high_risk_prob

        # --- Target 5: Multi-class Classification - Customer Segment ---
        # 3 segments: Standard, Premium, Elite (balanced: ~33% each)
        # FIXED: Ensure balanced classes with minimum samples

        # Generate segment scores with controlled bias
        segment_scores = np.column_stack([
            # Standard Segment (0)
            -0.5 +
            df['annual_income'] / 100000 * 0.3 +
            df['age'] / 90 * 0.2 +
            df['years_as_customer'] / 50 * 0.1,

            # Premium Segment (1)
            0.5 +
            df['annual_income'] / 100000 * 0.8 +
            df['savings_balance'] / 100000 * 0.5 +
            df['credit_score'] / 850 * 0.3 +
            (df['insurance_package'] == 'Premium').astype(float) * 0.5,

            # Elite Segment (2)
            1.0 +
            df['annual_income'] / 100000 * 1.2 +
            df['savings_balance'] / 100000 * 0.8 +
            df['policy_value'] / 1000000 * 0.5 +
            (df['insurance_package'] == 'Elite').astype(float) * 1.0 +
            (df['years_as_customer'] > 10).astype(float) * 0.3
        ])

        # Add country-specific bias with controlled values
        country_bias = {
            'Germany': [0, 0.1, 0.1],
            'France': [0.1, 0, 0.1],
            'Belgium': [0.2, 0, -0.1],
            'Switzerland': [-0.2, 0.1, 0.3]
        }
        segment_scores += country_bias[country]

        # Add noise
        segment_scores += np.random.normal(0, 0.3, segment_scores.shape)  # Reduced noise for stability

        # Use softmax to get probabilities
        exp_scores = np.exp(segment_scores - segment_scores.max(axis=1, keepdims=True))
        segment_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # Assign segments based on probabilities with balanced sampling
        # This ensures each class has at least 20% of samples
        segment_labels = np.argmax(segment_probs, axis=1)

        # Check class balance and adjust if needed
        unique, counts = np.unique(segment_labels, return_counts=True)
        min_samples = n * 0.15  # At least 15% per class

        # If any class has too few samples, redistribute
        if len(unique) < 3 or min(counts) < min_samples:
            # Use balanced random assignment with country-specific probabilities
            base_probs = {
                'Germany': [0.30, 0.35, 0.35],
                'France': [0.35, 0.30, 0.35],
                'Belgium': [0.40, 0.30, 0.30],
                'Switzerland': [0.25, 0.35, 0.40]
            }
            probs = base_probs.get(country, [0.33, 0.33, 0.34])
            segment_labels = np.random.choice([0, 1, 2], size=n, p=probs)

        # Convert to pandas Series
        targets['customer_segment'] = pd.Series(segment_labels, index=df.index)

        # Store segment probabilities
        targets['segment_probabilities'] = segment_probs.tolist()

        return targets

    def _add_data_quality_issues(self, df, country):
        """Add realistic data quality issues for enterprise data"""

        # 1. Missing Values (5-8% depending on field)
        missing_fields = ['annual_income', 'savings_balance', 'credit_score', 'medical_expenses']
        missing_rate = np.random.uniform(0.03, 0.07)

        for field in missing_fields:
            if np.random.random() < 0.3:  # 30% chance of field having missing
                mask = np.random.random(len(df)) < missing_rate
                df.loc[mask, field] = np.nan

        # 2. Outliers in critical fields (1-2%)
        outlier_fields = ['annual_income', 'premium_amount', 'policy_value']
        for field in outlier_fields:
            if np.random.random() < 0.2:
                n_outliers = int(len(df) * 0.01)
                idx = np.random.choice(len(df), n_outliers, replace=False)
                if field == 'annual_income':
                    df.loc[idx, field] = df[field].max() * np.random.uniform(2, 5, n_outliers)
                else:
                    df.loc[idx, field] = df[field].max() * np.random.uniform(3, 10, n_outliers)

        # 3. Inconsistent records (age vs experience)
        inconsistent_mask = np.random.random(len(df)) < 0.01
        df.loc[inconsistent_mask, 'driving_experience'] = df.loc[inconsistent_mask, 'age'] + 5

        # 4. Data entry errors (negative values in positive fields)
        error_mask = np.random.random(len(df)) < 0.001
        df.loc[error_mask, 'savings_balance'] = -df.loc[error_mask, 'savings_balance']

        return df

    def _create_centralized_dataset(self, all_client_data):
        """Create centralized dataset by combining all client data"""

        print("\n📦 Creating centralized dataset...")

        # Combine all client data
        centralized_train = pd.DataFrame()
        centralized_val = pd.DataFrame()
        centralized_test = pd.DataFrame()

        for client_id, splits in all_client_data.items():
            # Add client identifier
            train = splits['train'].copy()
            val = splits['val'].copy()
            test = splits['test'].copy()

            train['client_id'] = client_id
            val['client_id'] = client_id
            test['client_id'] = client_id

            centralized_train = pd.concat([centralized_train, train], ignore_index=True)
            centralized_val = pd.concat([centralized_val, val], ignore_index=True)
            centralized_test = pd.concat([centralized_test, test], ignore_index=True)

        # Shuffle centralized datasets
        centralized_train = centralized_train.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        centralized_val = centralized_val.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        centralized_test = centralized_test.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        print(f"  ✅ Centralized Train: {len(centralized_train):,} samples")
        print(f"  ✅ Centralized Val: {len(centralized_val):,} samples")
        print(f"  ✅ Centralized Test: {len(centralized_test):,} samples")

        return {
            'train': centralized_train,
            'val': centralized_val,
            'test': centralized_test
        }

    def generate_dataset(self, output_dir="data/synthetic/legacy_european_insurance_federated",
                        save_format='parquet', generate_centralized=True):
        """
        Generate complete federated dataset with train/val/test splits

        Parameters:
        -----------
        output_dir : str
            Directory to save the dataset (deterministic path)
            Default (non-canonical legacy path): "data/synthetic/legacy_european_insurance_federated"
        save_format : str
            'parquet' or 'npy' format for saving
            Default: 'parquet'
        generate_centralized : bool
            Whether to generate centralized dataset
            Default: True
        """

        countries = ['Germany', 'France', 'Belgium', 'Switzerland']
        all_data = {}

        print("="*80)
        print("🏦 European Insurance Federated Dataset Generation")
        print("="*80)
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Version: {self.dataset_version}")
        print(f"🔢 Random seed: {self.random_seed}")
        print(f"📝 Samples per client: {self.samples_per_client}")
        print(f"📦 Generate centralized: {generate_centralized}")
        print("="*80)

        # Create deterministic output directory using pathlib
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for country in countries:
            print(f"\n📊 Generating data for {country}...")

            # Generate data
            data = self._generate_client_data(country)
            data = self._add_data_quality_issues(data, country)

            # Generate customer IDs
            data['customer_id'] = [f"{country[:2]}{''.join([str(np.random.randint(0,9)) for _ in range(8)])}"
                                   for _ in range(len(data))]

            # Split into train/val/test (60/20/20)
            train_data, temp_data = train_test_split(
                data, test_size=0.4, random_state=self.random_seed, stratify=data['customer_segment']
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, random_state=self.random_seed, stratify=temp_data['customer_segment']
            )

            # Store
            client_id = f"{country.lower()}_{self.country_stats[country]['company'].lower()}"
            all_data[client_id] = {
                'train': train_data,
                'val': val_data,
                'test': test_data,
                'metadata': {
                    'country': country,
                    'company': self.country_stats[country]['company'],
                    'total_samples': len(data),
                    'train_samples': len(train_data),
                    'val_samples': len(val_data),
                    'test_samples': len(test_data),
                    'feature_count': len([col for col in data.columns
                                        if col not in ['customer_id', 'country', 'insurance_company',
                                                      'regulatory_framework', 'currency']]),
                    'target_distribution': {
                        'claim_risk_score': {'mean': float(data['claim_risk_score'].mean()),
                                            'std': float(data['claim_risk_score'].std())},
                        'customer_lifetime_value': {'mean': float(data['customer_lifetime_value'].mean()),
                                                   'std': float(data['customer_lifetime_value'].std())},
                        'will_renew': data['will_renew'].value_counts().to_dict(),
                        'is_high_risk': data['is_high_risk'].value_counts().to_dict(),
                        'customer_segment': data['customer_segment'].value_counts().to_dict()
                    }
                }
            }

            print(f"  ✅ Generated {len(data):,} samples")
            print(f"  📈 Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(test_data):,}")
            print(f"  🎯 Target distribution:")
            print(f"    - Claim Risk Score: {data['claim_risk_score'].mean():.2f} ± {data['claim_risk_score'].std():.2f}")
            print(f"    - Will Renew: {data['will_renew'].value_counts().to_dict()}")
            print(f"    - High Risk: {data['is_high_risk'].value_counts().to_dict()}")
            print(f"    - Segments: {data['customer_segment'].value_counts().to_dict()}")

        # Create centralized dataset if requested
        centralized_data = None
        if generate_centralized:
            centralized_data = self._create_centralized_dataset(all_data)

        # Save dataset
        self._save_dataset(all_data, output_dir, save_format, centralized_data)

        return all_data, centralized_data

    def _save_dataset(self, all_data, output_dir, save_format='parquet', centralized_data=None):
        """Save dataset in parquet or npy format with deterministic paths"""

        print(f"\n💾 Saving dataset to {output_dir}/")

        output_path = Path(output_dir)

        # Save federated data (client-specific)
        print("\n📁 Saving federated data...")
        for client_id, splits in all_data.items():
            client_dir = output_path / "federated" / client_id
            client_dir.mkdir(parents=True, exist_ok=True)

            for split_name, df in splits.items():
                if split_name == 'metadata':
                    continue

                if save_format == 'parquet':
                    # Save as Parquet (recommended for large datasets)
                    file_path = client_dir / f"{split_name}.parquet"
                    table = pa.Table.from_pandas(df)
                    pq.write_table(table, str(file_path))
                else:
                    # Save as NPY (for numerical data only)
                    # Convert categorical to numeric for NPY
                    df_numeric = df.select_dtypes(include=[np.number])
                    file_path = client_dir / f"{split_name}.npy"
                    np.save(str(file_path), df_numeric.values)

                    # Also save column names
                    cols_path = client_dir / f"{split_name}_columns.txt"
                    with open(cols_path, 'w') as f:
                        f.write('\n'.join(df_numeric.columns))

        # Save centralized data if provided
        if centralized_data is not None:
            print("\n📁 Saving centralized data...")
            centralized_dir = output_path / "centralized"
            centralized_dir.mkdir(parents=True, exist_ok=True)

            for split_name, df in centralized_data.items():
                if save_format == 'parquet':
                    file_path = centralized_dir / f"{split_name}.parquet"
                    table = pa.Table.from_pandas(df)
                    pq.write_table(table, str(file_path))
                else:
                    df_numeric = df.select_dtypes(include=[np.number])
                    file_path = centralized_dir / f"{split_name}.npy"
                    np.save(str(file_path), df_numeric.values)

        # Save global metadata with deterministic version
        metadata = {
            'dataset_name': 'European Insurance Federated Learning Dataset',
            'dataset_version': self.dataset_version,
            'random_seed': self.random_seed,
            'samples_per_client': self.samples_per_client,
            'generation_date': datetime.now().isoformat(),
            'clients': list(all_data.keys()),
            'countries': ['Germany', 'France', 'Belgium', 'Switzerland'],
            'companies': [self.country_stats[c]['company'] for c in ['Germany', 'France', 'Belgium', 'Switzerland']],
            'total_samples_per_client': self.samples_per_client,
            'total_samples': self.samples_per_client * len(all_data),
            'has_centralized_dataset': centralized_data is not None,
            'feature_types': {
                'numerical': self.feature_definitions['numerical'],
                'categorical': self.feature_definitions['categorical']
            },
            'targets': {
                'claim_risk_score': {'type': 'regression', 'range': [0, 100]},
                'customer_lifetime_value': {'type': 'regression', 'range': [1000, 250000]},
                'will_renew': {'type': 'binary', 'classes': ['No', 'Yes']},
                'is_high_risk': {'type': 'binary', 'classes': ['No', 'Yes']},
                'customer_segment': {'type': 'multi_class', 'classes': ['Standard', 'Premium', 'Elite']}
            },
            'country_statistics': self.country_stats,
            'splits': {'train': 0.6, 'val': 0.2, 'test': 0.2}
        }

        # Save client-specific metadata
        for client_id, splits in all_data.items():
            if 'metadata' in splits:
                metadata[f'client_{client_id}_metadata'] = splits['metadata']

        # Add centralized dataset statistics if available
        if centralized_data is not None:
            metadata['centralized'] = {
                'train_samples': len(centralized_data['train']),
                'val_samples': len(centralized_data['val']),
                'test_samples': len(centralized_data['test']),
                'total_samples': len(centralized_data['train']) + len(centralized_data['val']) + len(centralized_data['test'])
            }

        # Save global metadata
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save dataset info
        info = {
            'dataset_version': self.dataset_version,
            'output_directory': str(output_dir),
            'save_format': save_format,
            'total_clients': len(all_data),
            'samples_per_client': self.samples_per_client,
            'total_samples': self.samples_per_client * len(all_data),
            'features_count': len(self.feature_definitions['numerical']) + len(self.feature_definitions['categorical']),
            'targets_count': 5,
            'has_centralized_dataset': centralized_data is not None,
            'file_structure': {
                'federated': {
                    client_id: {
                        'train': f"federated/{client_id}/train.{save_format}",
                        'val': f"federated/{client_id}/val.{save_format}",
                        'test': f"federated/{client_id}/test.{save_format}"
                    } for client_id in all_data.keys()
                }
            }
        }

        if centralized_data is not None:
            info['file_structure']['centralized'] = {
                'train': f"centralized/train.{save_format}",
                'val': f"centralized/val.{save_format}",
                'test': f"centralized/test.{save_format}"
            }

        info_path = output_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2, default=str)

        print(f"\n✅ Dataset saved successfully!")
        print(f"📁 Location: {output_dir}/")
        print(f"📄 Metadata: {output_path / 'metadata.json'}")
        print(f"📄 Dataset Info: {output_path / 'dataset_info.json'}")

        return output_dir

    def load_dataset(self, dataset_path, load_centralized=True):
        """Load saved dataset with deterministic paths"""
        all_data = {}

        dataset_path = Path(dataset_path)

        # Load metadata
        metadata_path = dataset_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load each client's data
        for client_id in metadata['clients']:
            client_dir = dataset_path / "federated" / client_id
            all_data[client_id] = {}

            for split in ['train', 'val', 'test']:
                parquet_path = client_dir / f"{split}.parquet"
                if parquet_path.exists():
                    all_data[client_id][split] = pd.read_parquet(str(parquet_path))
                else:
                    # Try loading NPY
                    npy_path = client_dir / f"{split}.npy"
                    if npy_path.exists():
                        all_data[client_id][split] = np.load(str(npy_path))

        # Load centralized data if requested and available
        centralized_data = None
        if load_centralized and metadata.get('has_centralized_dataset', False):
            centralized_dir = dataset_path / "centralized"
            centralized_data = {}
            for split in ['train', 'val', 'test']:
                parquet_path = centralized_dir / f"{split}.parquet"
                if parquet_path.exists():
                    centralized_data[split] = pd.read_parquet(str(parquet_path))
                else:
                    npy_path = centralized_dir / f"{split}.npy"
                    if npy_path.exists():
                        centralized_data[split] = np.load(str(npy_path))

        return all_data, centralized_data, metadata

    def get_dataset_summary(self, dataset_path):
        """Get a summary of the dataset"""
        summary = {}

        dataset_path = Path(dataset_path)

        # Load metadata
        metadata_path = dataset_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        summary['metadata'] = metadata

        # Load and analyze a sample client
        client_id = metadata['clients'][0]
        train_path = dataset_path / "federated" / client_id / "train.parquet"
        if train_path.exists():
            df = pd.read_parquet(str(train_path))
            summary['sample_features'] = df.columns.tolist()
            summary['sample_shape'] = df.shape
            summary['numeric_features'] = df.select_dtypes(include=[np.number]).columns.tolist()
            summary['categorical_features'] = df.select_dtypes(include=['object']).columns.tolist()
            summary['target_columns'] = ['claim_risk_score', 'customer_lifetime_value',
                                        'will_renew', 'is_high_risk', 'customer_segment']

        # Load centralized summary if available
        if metadata.get('has_centralized_dataset', False):
            centralized_dir = dataset_path / "centralized"
            train_path = centralized_dir / "train.parquet"
            if train_path.exists():
                df = pd.read_parquet(str(train_path))
                summary['centralized_shape'] = df.shape
                summary['centralized_samples'] = len(df)

        return summary


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":

    print("🏦 European Insurance Federated Dataset Generator")
    print("="*80)

    # Generate dataset with deterministic paths
    generator = EuropeanInsuranceFederatedDataset(
        samples_per_client=5000,
        random_seed=42,
        dataset_version="v1.0"
    )

    # Generate and save with fixed path (using project structure)
    output_dir = "data/synthetic/legacy_european_insurance_federated"
    federated_data, centralized_data = generator.generate_dataset(
        output_dir=output_dir,
        save_format='parquet',
        generate_centralized=True
    )

    # Print sample data
    print("\n" + "="*80)
    print("📊 Sample Data from Germany (Allianz)")
    print("="*80)

    client_id = 'germany_allianz'
    print(f"\nClient: {client_id}")
    print(f"Train shape: {federated_data[client_id]['train'].shape}")
    print(f"Val shape: {federated_data[client_id]['val'].shape}")
    print(f"Test shape: {federated_data[client_id]['test'].shape}")

    print("\n🔍 First 5 rows:")
    print(federated_data[client_id]['train'].head())

    print("\n📈 Column descriptions:")
    for col in federated_data[client_id]['train'].columns[:10]:
        if federated_data[client_id]['train'][col].dtype in ['float64', 'int64']:
            print(f"  {col}: mean={federated_data[client_id]['train'][col].mean():.2f}, "
                  f"min={federated_data[client_id]['train'][col].min():.2f}, "
                  f"max={federated_data[client_id]['train'][col].max():.2f}")
        else:
            print(f"  {col}: unique values={federated_data[client_id]['train'][col].nunique()}")

    print("\n🎯 Target distributions:")
    for target in ['claim_risk_score', 'customer_lifetime_value',
                   'will_renew', 'is_high_risk', 'customer_segment']:
        if target in ['claim_risk_score', 'customer_lifetime_value']:
            print(f"  {target}: mean={federated_data[client_id]['train'][target].mean():.2f}")
        else:
            print(f"  {target}:\n    {federated_data[client_id]['train'][target].value_counts().to_dict()}")

    # Print centralized dataset info
    if centralized_data is not None:
        print("\n" + "="*80)
        print("📦 Centralized Dataset Information")
        print("="*80)
        print(f"Train shape: {centralized_data['train'].shape}")
        print(f"Val shape: {centralized_data['val'].shape}")
        print(f"Test shape: {centralized_data['test'].shape}")

        # Show client distribution in centralized dataset
        print("\nClient distribution in centralized train set:")
        print(centralized_data['train']['client_id'].value_counts().to_dict())

    # Get dataset summary
    print("\n" + "="*80)
    print("📊 Dataset Summary")
    print("="*80)
    summary = generator.get_dataset_summary(output_dir)
    print(f"Total clients: {len(summary['metadata']['clients'])}")
    print(f"Total samples: {summary['metadata']['samples_per_client'] * len(summary['metadata']['clients'])}")
    print(f"Features: {len(summary['sample_features'])}")
    print(f"Sample shape: {summary['sample_shape']}")
    if 'centralized_shape' in summary:
        print(f"Centralized samples: {summary['centralized_samples']}")

    print("\n✅ Dataset generation complete!")
    print(f"📁 Dataset location: {output_dir}/")
    print(f"📄 Metadata: {output_dir}/metadata.json")
    print(f"📄 Dataset info: {output_dir}/dataset_info.json")

    # Show folder structure
    print("\n📁 Generated folder structure:")
    print("data/")
    print("└── processed/")
    print("    └── european_insurance_federated/")
    print("        ├── federated/")
    for client in summary['metadata']['clients']:
        print(f"        │   ├── {client}/")
        print(f"        │   │   ├── train.parquet")
        print(f"        │   │   ├── val.parquet")
        print(f"        │   │   └── test.parquet")
    print("        ├── centralized/")
    print("        │   ├── train.parquet")
    print("        │   ├── val.parquet")
    print("        │   └── test.parquet")
    print("        ├── metadata.json")
    print("        └── dataset_info.json")