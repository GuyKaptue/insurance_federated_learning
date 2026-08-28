"""
Data Schema Definition for Insurance Dataset
Defines column types, constraints, and metadata
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ColumnType(Enum):
    """Column type enumeration"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TARGET = "target"
    METADATA = "metadata"
    ID = "id"


class TargetType(Enum):
    """Target type enumeration"""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASSIFICATION = "multi_classification"


@dataclass
class ColumnMetadata:
    """Metadata for a single column"""
    name: str
    column_type: ColumnType
    data_type: str
    description: str = ""
    is_nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[str]] = None
    target_type: Optional[TargetType] = None
    unit: str = ""
    transformation: Optional[str] = None


@dataclass
class DataSchema:
    """Complete data schema for insurance dataset"""
    version: str = "1.0.0"
    columns: List[ColumnMetadata] = field(default_factory=list)

    def get_column_names(self) -> List[str]:
        """Get all column names"""
        return [col.name for col in self.columns]

    def get_numerical_columns(self) -> List[str]:
        """Get numerical column names"""
        return [col.name for col in self.columns if col.column_type == ColumnType.NUMERICAL]

    def get_categorical_columns(self) -> List[str]:
        """Get categorical column names"""
        return [col.name for col in self.columns if col.column_type == ColumnType.CATEGORICAL]

    def get_target_columns(self) -> List[str]:
        """Get target column names"""
        return [col.name for col in self.columns if col.column_type == ColumnType.TARGET]

    def get_metadata_columns(self) -> List[str]:
        """Get metadata column names"""
        return [col.name for col in self.columns if col.column_type == ColumnType.METADATA]

    def get_id_columns(self) -> List[str]:
        """Get ID column names"""
        return [col.name for col in self.columns if col.column_type == ColumnType.ID]

    def get_column_by_name(self, name: str) -> Optional[ColumnMetadata]:
        """Get column metadata by name"""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary"""
        return {
            'version': self.version,
            'columns': [
                {
                    'name': col.name,
                    'column_type': col.column_type.value,
                    'data_type': col.data_type,
                    'description': col.description,
                    'is_nullable': col.is_nullable,
                    'min_value': col.min_value,
                    'max_value': col.max_value,
                    'categories': col.categories,
                    'target_type': col.target_type.value if col.target_type else None,
                    'unit': col.unit,
                    'transformation': col.transformation
                }
                for col in self.columns
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataSchema':
        """Create schema from dictionary"""
        schema = cls(version=data.get('version', '1.0.0'))

        for col_data in data.get('columns', []):
            column = ColumnMetadata(
                name=col_data['name'],
                column_type=ColumnType(col_data['column_type']),
                data_type=col_data['data_type'],
                description=col_data.get('description', ''),
                is_nullable=col_data.get('is_nullable', True),
                min_value=col_data.get('min_value'),
                max_value=col_data.get('max_value'),
                categories=col_data.get('categories'),
                target_type=TargetType(col_data['target_type']) if col_data.get('target_type') else None,
                unit=col_data.get('unit', ''),
                transformation=col_data.get('transformation')
            )
            schema.columns.append(column)

        return schema


def get_insurance_schema() -> DataSchema:
    """Get the default insurance dataset schema"""

    schema = DataSchema(version="1.0.0")

    # Numerical features
    numerical_columns = [
        ('annual_income', 'Annual income of the customer', 'EUR/CHF', (15000, 300000)),
        ('savings_balance', 'Total savings balance', 'EUR/CHF', (0, 200000)),
        ('policy_count', 'Number of insurance policies', 'count', (1, 12)),
        ('policy_value', 'Total policy value', 'EUR/CHF', (5000, 5000000)),
        ('age', 'Customer age', 'years', (18, 95)),
        ('years_as_customer', 'Years as customer', 'years', (0, 50)),
        ('number_of_claims', 'Number of claims filed', 'count', (0, 20)),
        ('total_claim_amount', 'Total claim amount', 'EUR/CHF', (0, 500000)),
        ('claim_frequency', 'Claim frequency per year', 'claims/year', (0, 0.5)),
        ('property_value', 'Property value', 'EUR/CHF', (100000, 5000000)),
        ('medical_expenses', 'Annual medical expenses', 'EUR/CHF', (0, 150000)),
        ('life_expectancy', 'Life expectancy', 'years', (60, 95)),
        ('health_risk_score', 'Health risk assessment score', 'score', (0, 100)),
        ('driving_experience', 'Years of driving experience', 'years', (0, 60)),
        ('vehicle_value', 'Vehicle value', 'EUR/CHF', (5000, 200000)),
        ('liability_risk', 'Liability risk score', 'score', (0, 100)),
        ('debt_to_income_ratio', 'Debt-to-income ratio', 'ratio', (0.05, 0.8)),
        ('credit_score', 'Credit score', 'score', (300, 850)),
        ('family_members', 'Number of family members', 'count', (1, 8)),
        ('policy_duration', 'Policy duration in years', 'years', (1, 30)),
        ('premium_amount', 'Annual premium amount', 'EUR/CHF', (200, 20000))
    ]

    for name, desc, unit, (min_val, max_val) in numerical_columns:
        schema.columns.append(ColumnMetadata(
            name=name,
            column_type=ColumnType.NUMERICAL,
            data_type='float64',
            description=desc,
            min_value=min_val,
            max_value=max_val,
            unit=unit,
            is_nullable=True
        ))

    # Categorical features
    categorical_columns = [
        ('marital_status', 'Marital status', ['Single', 'Married', 'Divorced', 'Widowed']),
        ('education_level', 'Education level', ['Secondary', 'Bachelor', 'Master', 'PhD', 'Professional']),
        ('employment_status', 'Employment status', ['Employed', 'Self-Employed', 'Unemployed', 'Retired']),
        ('home_ownership', 'Home ownership status', ['Own', 'Rent', 'Mortgage', 'Other']),
        ('insurance_package', 'Insurance package type', ['Basic', 'Standard', 'Premium', 'Elite']),
        ('risk_category', 'Risk category', ['Low', 'Medium', 'High', 'Very High']),
        ('policy_type', 'Policy type', ['Individual', 'Family', 'Corporate']),
        ('vehicle_type', 'Vehicle type', ['Sedan', 'SUV', 'Sports', 'Truck', 'Motorcycle']),
        ('property_type', 'Property type', ['Apartment', 'House', 'Villa', 'Commercial']),
        ('profession', 'Profession', ['Engineer', 'Doctor', 'Lawyer', 'Teacher', 'Business', 'Other']),
        ('primary_insurance_type', 'Primary insurance type', ['Auto', 'Life', 'Health', 'Property', 'Liability', 'Travel', 'Accident']),
        ('policy_status', 'Policy status', ['Active', 'Lapsed', 'Cancelled', 'Pending'])
    ]

    for name, desc, categories in categorical_columns:
        schema.columns.append(ColumnMetadata(
            name=name,
            column_type=ColumnType.CATEGORICAL,
            data_type='object',
            description=desc,
            categories=categories,
            is_nullable=True
        ))

    # Target columns
    target_columns = [
        ('claim_risk_score', 'Claim risk score (0-100)', TargetType.REGRESSION, 'score'),
        ('customer_lifetime_value', 'Customer lifetime value', TargetType.REGRESSION, 'EUR/CHF'),
        ('will_renew', 'Will renew policy', TargetType.BINARY_CLASSIFICATION, 'binary'),
        ('is_high_risk', 'High risk customer', TargetType.BINARY_CLASSIFICATION, 'binary'),
        ('customer_segment', 'Customer segment', TargetType.MULTI_CLASSIFICATION, 'segment')
    ]

    for name, desc, target_type, unit in target_columns:
        schema.columns.append(ColumnMetadata(
            name=name,
            column_type=ColumnType.TARGET,
            data_type='float64' if target_type == TargetType.REGRESSION else 'int64',
            description=desc,
            target_type=target_type,
            unit=unit,
            is_nullable=False
        ))

    # Metadata columns
    metadata_columns = [
        ('country', 'Country of the customer', 'string'),
        ('insurance_company', 'Insurance company name', 'string'),
        ('regulatory_framework', 'Regulatory framework', 'string'),
        ('currency', 'Currency', 'string')
    ]

    for name, desc, data_type in metadata_columns:
        schema.columns.append(ColumnMetadata(
            name=name,
            column_type=ColumnType.METADATA,
            data_type='object',
            description=desc,
            is_nullable=False
        ))

    # ID columns
    schema.columns.append(ColumnMetadata(
        name='customer_id',
        column_type=ColumnType.ID,
        data_type='object',
        description='Unique customer identifier',
        is_nullable=False
    ))

    return schema