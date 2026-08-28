"""
Data Quality Analyzer for European Insurance Federated Learning
Comprehensive data quality analysis and reporting
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.core.loader.data_schema import DataSchema, get_insurance_schema, ColumnType


@dataclass
class QualityMetrics:
    """Container for data quality metrics"""
    total_rows: int
    total_columns: int
    missing_values: Dict[str, float]  # Column -> percentage missing
    duplicate_rows: int
    duplicate_percentage: float
    outlier_counts: Dict[str, int]  # Column -> number of outliers
    outlier_percentages: Dict[str, float]  # Column -> percentage outliers
    unique_values: Dict[str, int]  # Column -> number of unique values
    value_counts: Dict[str, Dict]  # Column -> value distribution
    column_types: Dict[str, str]  # Column -> data type
    correlation_matrix: Optional[pd.DataFrame] = None
    summary_stats: Dict[str, Dict] = field(default_factory=dict)  # Column -> statistics


class DataQualityAnalyzer:
    """
    Comprehensive Data Quality Analyzer
    Analyzes data quality issues, distributions, and patterns
    """

    def __init__(self, schema: Optional[DataSchema] = None):
        """
        Initialize DataQualityAnalyzer

        Parameters:
        -----------
        schema : Optional[DataSchema]
            Data schema for validation (default: uses insurance schema)
        """
        self.schema = schema or get_insurance_schema()
        self.quality_metrics = None
        self.reports = {}

    def _is_hashable_column(self, col: pd.Series) -> bool:
        """Check if a column contains hashable values"""
        try:
            # Try to get first non-null value to check type
            sample = col.dropna().iloc[0] if len(col.dropna()) > 0 else None
            if sample is None:
                return True
            # Check if it's a list, array, or dict (unhashable)
            if isinstance(sample, (list, np.ndarray, dict)):
                return False
            # Try to hash a sample value
            hash(sample)
            return True
        except (TypeError, ValueError, AttributeError):
            return False

    def _prepare_for_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for duplicate detection by converting unhashable types"""
        df_copy = df.copy()

        for col in df_copy.columns:
            # Skip if column is all NaN
            if df_copy[col].isnull().all():
                continue

            # Check if column contains unhashable types
            if not self._is_hashable_column(df_copy[col]):
                # Convert to string representation
                df_copy[col] = df_copy[col].apply(
                    lambda x: str(x) if x is not None and not isinstance(x, (int, float, str, bool)) else x
                )

        return df_copy

    def analyze(self, df: pd.DataFrame, target_columns: Optional[List[str]] = None) -> QualityMetrics:
        """
        Perform comprehensive data quality analysis

        Parameters:
        -----------
        df : pd.DataFrame
            Data to analyze
        target_columns : Optional[List[str]]
            List of target column names (uses schema targets if None)

        Returns:
        --------
        QualityMetrics
            Quality metrics container
        """
        if target_columns is None:
            target_columns = self.schema.get_target_columns()

        # Basic metrics
        total_rows = len(df)
        total_columns = len(df.columns)

        # Missing values
        missing_values = {}
        for col in df.columns:
            missing_pct = (df[col].isnull().sum() / total_rows) * 100
            if missing_pct > 0:
                missing_values[col] = missing_pct

        # Duplicate rows - use prepared dataframe
        df_for_duplicates = self._prepare_for_duplicates(df)
        duplicate_rows = df_for_duplicates.duplicated().sum()
        duplicate_pct = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0

        # Column types
        column_types = df.dtypes.astype(str).to_dict()

        # Unique values and value counts
        unique_values = {}
        value_counts = {}
        for col in df.columns:
            try:
                unique_values[col] = df[col].nunique()

                # Only compute value counts for hashable columns with reasonable cardinality
                if self._is_hashable_column(df[col]) and (df[col].dtype == 'object' or df[col].nunique() < 50):
                    # For object columns or low cardinality, get value counts
                    vc = df[col].value_counts()
                    # Convert to dict, handling any unhashable types
                    value_counts[col] = {str(k): int(v) for k, v in vc.items()}
                else:
                    value_counts[col] = {}
            except Exception as e:
                # If any error, skip value counts for this column
                unique_values[col] = 0
                value_counts[col] = {}

        # Outliers (for numerical columns)
        outlier_counts = {}
        outlier_percentages = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            # Skip if column has all NaN
            if df[col].isnull().all():
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
            outlier_percentages[col] = (len(outliers) / total_rows) * 100

        # Summary statistics
        summary_stats = {}
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # Skip if column has all NaN
                if df[col].isnull().all():
                    continue

                try:
                    stats_dict = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        '25%': df[col].quantile(0.25),
                        '50%': df[col].median(),
                        '75%': df[col].quantile(0.75),
                        'max': df[col].max(),
                        'skew': df[col].skew(),
                        'kurtosis': df[col].kurtosis()
                    }
                    summary_stats[col] = stats_dict
                except Exception as e:
                    # Skip columns that cause issues
                    continue

        # Correlation matrix (for numerical columns)
        correlation_matrix = None
        if len(numerical_cols) > 1:
            try:
                # Filter out columns with all NaN
                valid_num_cols = [col for col in numerical_cols if not df[col].isnull().all()]
                if len(valid_num_cols) > 1:
                    correlation_matrix = df[valid_num_cols].corr()
            except Exception as e:
                # Skip correlation if error
                pass

        # Create QualityMetrics object
        self.quality_metrics = QualityMetrics(
            total_rows=total_rows,
            total_columns=total_columns,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            duplicate_percentage=duplicate_pct,
            outlier_counts=outlier_counts,
            outlier_percentages=outlier_percentages,
            unique_values=unique_values,
            value_counts=value_counts,
            column_types=column_types,
            correlation_matrix=correlation_matrix,
            summary_stats=summary_stats
        )

        return self.quality_metrics

    def analyze_client(self, data_dict: Dict[str, pd.DataFrame], client_id: str) -> QualityMetrics:
        """
        Analyze data for a specific client

        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary containing 'train', 'val', 'test' dataframes
        client_id : str
            Client identifier

        Returns:
        --------
        QualityMetrics
            Quality metrics for the client
        """
        # Combine all splits for analysis
        combined_df = pd.concat([
            data_dict.get('train', pd.DataFrame()),
            data_dict.get('val', pd.DataFrame()),
            data_dict.get('test', pd.DataFrame())
        ], ignore_index=True)

        if len(combined_df) == 0:
            raise ValueError(f"No data found for client {client_id}")

        return self.analyze(combined_df)

    def generate_report(self, df: pd.DataFrame, output_dir: Optional[str] = None) -> Dict:
        """
        Generate comprehensive quality report

        Parameters:
        -----------
        df : pd.DataFrame
            Data to analyze
        output_dir : Optional[str]
            Directory to save report (if provided)

        Returns:
        --------
        Dict
            Report dictionary
        """
        if self.quality_metrics is None:
            self.analyze(df)

        report = {
            'data_quality_summary': self._get_quality_summary(),
            'missing_values': self._format_missing_values(),
            'outliers': self._format_outliers(),
            'column_statistics': self._format_column_stats(),
            'target_analysis': self._analyze_targets(df),
            'recommendations': self._get_recommendations()
        }

        self.reports['quality_report'] = report

        # Save report if output_dir provided
        if output_dir:
            self._save_report(report, output_dir)

        return report

    def generate_client_report(self, data_dict: Dict[str, pd.DataFrame],
                              client_id: str, output_dir: Optional[str] = None) -> Dict:
        """
        Generate report for a specific client

        Parameters:
        -----------
        data_dict : Dict[str, pd.DataFrame]
            Dictionary containing 'train', 'val', 'test' dataframes
        client_id : str
            Client identifier
        output_dir : Optional[str]
            Directory to save report

        Returns:
        --------
        Dict
            Client report dictionary
        """
        self.analyze_client(data_dict, client_id)
        combined_df = pd.concat([
            data_dict.get('train', pd.DataFrame()),
            data_dict.get('val', pd.DataFrame()),
            data_dict.get('test', pd.DataFrame())
        ], ignore_index=True)

        report = self.generate_report(combined_df)
        report['client_id'] = client_id

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save as JSON with deterministic filename
            report_file = output_path / f"quality_report_{client_id}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        return report

    def _get_quality_summary(self) -> Dict:
        """Get summary of data quality"""
        return {
            'total_samples': self.quality_metrics.total_rows,
            'total_features': self.quality_metrics.total_columns,
            'missing_value_count': len(self.quality_metrics.missing_values),
            'missing_value_total': sum(self.quality_metrics.missing_values.values()),
            'duplicate_percentage': self.quality_metrics.duplicate_percentage,
            'outlier_count': sum(self.quality_metrics.outlier_counts.values()),
            'overall_quality_score': self._calculate_quality_score()
        }

    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0

        # Penalize missing values
        if self.quality_metrics.missing_values:
            avg_missing = sum(self.quality_metrics.missing_values.values()) / len(self.quality_metrics.missing_values)
            score -= min(avg_missing * 2, 30)

        # Penalize duplicates
        score -= min(self.quality_metrics.duplicate_percentage * 0.5, 10)

        # Penalize outliers
        if self.quality_metrics.outlier_percentages:
            avg_outlier = sum(self.quality_metrics.outlier_percentages.values()) / len(self.quality_metrics.outlier_percentages)
            score -= min(avg_outlier * 0.3, 20)

        return max(0, score)

    def _format_missing_values(self) -> Dict:
        """Format missing values information"""
        if not self.quality_metrics.missing_values:
            return {'message': 'No missing values found'}

        sorted_missing = sorted(
            self.quality_metrics.missing_values.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'columns_with_missing': len(self.quality_metrics.missing_values),
            'missing_columns': sorted_missing,
            'total_missing_percentage': sum(self.quality_metrics.missing_values.values())
        }

    def _format_outliers(self) -> Dict:
        """Format outlier information"""
        if not self.quality_metrics.outlier_counts:
            return {'message': 'No outliers found'}

        sorted_outliers = sorted(
            self.quality_metrics.outlier_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'columns_with_outliers': len(self.quality_metrics.outlier_counts),
            'outlier_columns': sorted_outliers
        }

    def _format_column_stats(self) -> Dict:
        """Format column statistics"""
        return {
            'numerical_columns': {
                col: stats for col, stats in self.quality_metrics.summary_stats.items()
                if col in self.schema.get_numerical_columns()
            },
            'categorical_columns': {
                col: {
                    'unique_values': self.quality_metrics.unique_values.get(col, 0),
                    'value_counts': self.quality_metrics.value_counts.get(col, {})
                }
                for col in self.schema.get_categorical_columns()
                if col in self.quality_metrics.unique_values
            }
        }

    def _analyze_targets(self, df: pd.DataFrame) -> Dict:
        """Analyze target columns"""
        target_cols = self.schema.get_target_columns()
        target_analysis = {}

        for target in target_cols:
            if target not in df.columns:
                continue

            target_analysis[target] = {
                'type': self._get_target_type(target),
                'distribution': self._get_target_distribution(df[target]),
                'class_balance': self._check_class_balance(df[target]) if df[target].dtype in ['int64', 'object'] else None
            }

        return target_analysis

    def _get_target_type(self, target: str) -> str:
        """Get target type from schema"""
        for col in self.schema.columns:
            if col.name == target:
                return col.target_type.value if col.target_type else 'unknown'
        return 'unknown'

    def _get_target_distribution(self, series: pd.Series) -> Dict:
        """Get target distribution"""
        # Remove NaN values for distribution calculation
        clean_series = series.dropna()

        if len(clean_series) == 0:
            return {'message': 'No valid values'}

        if series.dtype in ['float64', 'int64'] and series.nunique() > 20:
            # Numerical - show percentiles
            return {
                'min': float(clean_series.min()),
                'max': float(clean_series.max()),
                'mean': float(clean_series.mean()),
                'median': float(clean_series.median()),
                'std': float(clean_series.std()),
                'percentiles': {
                    '25%': float(clean_series.quantile(0.25)),
                    '50%': float(clean_series.quantile(0.50)),
                    '75%': float(clean_series.quantile(0.75)),
                    '90%': float(clean_series.quantile(0.90)),
                    '95%': float(clean_series.quantile(0.95))
                }
            }
        else:
            # Categorical - show counts
            try:
                vc = clean_series.value_counts()
                return {
                    'unique_count': int(clean_series.nunique()),
                    'value_counts': {str(k): int(v) for k, v in vc.items()},
                    'mode': str(clean_series.mode().iloc[0]) if not clean_series.empty else None
                }
            except Exception as e:
                return {
                    'unique_count': int(clean_series.nunique()),
                    'value_counts': {},
                    'mode': None,
                    'error': str(e)
                }

    def _check_class_balance(self, series: pd.Series) -> Dict:
        """Check class balance"""
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {'message': 'No valid values'}

        counts = clean_series.value_counts()
        total = len(clean_series)

        balance = {}
        for cls, count in counts.items():
            try:
                balance[str(cls)] = {
                    'count': int(count),
                    'percentage': (count / total) * 100
                }
            except:
                balance[str(cls)] = {
                    'count': int(count),
                    'percentage': (count / total) * 100
                }

        # Calculate imbalance ratio
        if len(counts) > 1:
            max_pct = max(counts / total)
            min_pct = min(counts / total)
            balance['imbalance_ratio'] = float(max_pct / min_pct) if min_pct > 0 else float('inf')
            balance['is_balanced'] = balance['imbalance_ratio'] < 2.0

        return balance

    def _get_recommendations(self) -> List[str]:
        """Get data quality recommendations"""
        recommendations = []

        # Missing values recommendations
        if self.quality_metrics.missing_values:
            rec = f"Consider imputing missing values in: {', '.join(list(self.quality_metrics.missing_values.keys())[:5])}"
            if len(self.quality_metrics.missing_values) > 5:
                rec += f" and {len(self.quality_metrics.missing_values) - 5} more"
            recommendations.append(rec)

        # Outlier recommendations
        if self.quality_metrics.outlier_counts:
            high_outlier_cols = [
                col for col, count in self.quality_metrics.outlier_counts.items()
                if count > self.quality_metrics.total_rows * 0.05
            ]
            if high_outlier_cols:
                rec = f"Review outliers in: {', '.join(high_outlier_cols[:3])}"
                if len(high_outlier_cols) > 3:
                    rec += f" and {len(high_outlier_cols) - 3} more"
                recommendations.append(rec)

        # Duplicate recommendations
        if self.quality_metrics.duplicate_percentage > 5:
            recommendations.append(
                f"Remove duplicate rows ({self.quality_metrics.duplicate_percentage:.2f}% duplicates)"
            )

        # Skewness recommendations
        skew_cols = []
        for col, stats in self.quality_metrics.summary_stats.items():
            if abs(stats.get('skew', 0)) > 2:
                skew_cols.append(col)
        if skew_cols:
            rec = f"Consider transformation for skewed columns: {', '.join(skew_cols[:3])}"
            if len(skew_cols) > 3:
                rec += f" and {len(skew_cols) - 3} more"
            recommendations.append(rec)

        # Correlation recommendations
        if self.quality_metrics.correlation_matrix is not None:
            corr_matrix = self.quality_metrics.correlation_matrix
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j]))

            if high_corr:
                rec = f"Remove highly correlated features: {', '.join([f'{a}-{b}' for a,b in high_corr[:3]])}"
                if len(high_corr) > 3:
                    rec += f" and {len(high_corr) - 3} more"
                recommendations.append(rec)

        if not recommendations:
            recommendations.append("Data quality looks good! No critical issues found.")

        return recommendations

    def _save_report(self, report: Dict, output_dir: str):
        """Save report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_path = output_path / "quality_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate HTML report
        try:
            html_path = output_path / "quality_report.html"
            self._generate_html_report(report, html_path)
        except Exception as e:
            print(f"Warning: Could not generate HTML report: {e}")

    def _generate_html_report(self, report: Dict, output_path: Path):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
                .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
                .label {{ font-size: 12px; color: #7f8c8d; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .recommendations {{ background: #fef9e7; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>📊 Data Quality Report</h1>

            <div class="metric">
                <div class="value">{report['data_quality_summary']['total_samples']:,}</div>
                <div class="label">Total Samples</div>
            </div>
            <div class="metric">
                <div class="value">{report['data_quality_summary']['total_features']}</div>
                <div class="label">Total Features</div>
            </div>
            <div class="metric">
                <div class="value">{report['data_quality_summary']['overall_quality_score']:.1f}</div>
                <div class="label">Quality Score (0-100)</div>
            </div>

            <h2>📋 Recommendations</h2>
            <div class="recommendations">
                <ul>
        """

        for rec in report.get('recommendations', []):
            html_content += f"<li>{rec}</li>"

        html_content += """
                </ul>
            </div>

            <h2>🔍 Missing Values</h2>
        """

        if report.get('missing_values', {}).get('message'):
            html_content += f"<p>{report['missing_values']['message']}</p>"
        else:
            html_content += "<table><tr><th>Column</th><th>Missing %</th></tr>"
            for col, pct in report.get('missing_values', {}).get('missing_columns', []):
                html_content += f"<tr><td>{col}</td><td>{pct:.2f}%</td></tr>"
            html_content += "</table>"

        html_content += """
            <h2>📈 Outliers</h2>
        """

        if report.get('outliers', {}).get('message'):
            html_content += f"<p>{report['outliers']['message']}</p>"
        else:
            html_content += "<table><tr><th>Column</th><th>Outlier Count</th></tr>"
            for col, count in report.get('outliers', {}).get('outlier_columns', []):
                html_content += f"<tr><td>{col}</td><td>{count}</td></tr>"
            html_content += "</table>"

        html_content += """
            <h2>🎯 Target Analysis</h2>
        """

        for target, analysis in report.get('target_analysis', {}).items():
            html_content += f"<h3>{target}</h3>"
            html_content += f"<p>Type: {analysis['type']}</p>"

            dist = analysis.get('distribution', {})
            if 'message' in dist:
                html_content += f"<p>Distribution: {dist['message']}</p>"
            elif 'mean' in dist:
                html_content += f"<p>Mean: {dist['mean']:.2f}, Std: {dist['std']:.2f}</p>"
                html_content += f"<p>Min: {dist['min']:.2f}, Max: {dist['max']:.2f}</p>"
            elif 'value_counts' in dist:
                html_content += "<p>Value Counts:</p><ul>"
                for cls, count in dist.get('value_counts', {}).items():
                    html_content += f"<li>{cls}: {count}</li>"
                html_content += "</ul>"

            if analysis.get('class_balance'):
                html_content += "<p>Class Balance:</p><ul>"
                for cls, info in analysis['class_balance'].items():
                    if cls not in ['imbalance_ratio', 'is_balanced']:
                        html_content += f"<li>{cls}: {info['count']} ({info['percentage']:.1f}%)</li>"
                if 'imbalance_ratio' in analysis['class_balance']:
                    html_content += f"<li>Imbalance Ratio: {analysis['class_balance']['imbalance_ratio']:.2f}</li>"
                if 'is_balanced' in analysis['class_balance']:
                    html_content += f"<li>Is Balanced: {analysis['class_balance']['is_balanced']}</li>"
                html_content += "</ul>"

        html_content += """
            <hr>
            <p style="color: #7f8c8d; font-size: 12px;">
                Report generated by Insurance Federated Learning Data Quality Analyzer
            </p>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

    def plot_quality_metrics(self, figsize=(15, 10)):
        """
        Plot data quality metrics

        Parameters:
        -----------
        figsize : tuple
            Figure size
        """
        if self.quality_metrics is None:
            raise ValueError("No quality metrics available. Run analyze() first.")

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Missing values
        if self.quality_metrics.missing_values:
            ax = axes[0, 0]
            missing_df = pd.DataFrame({
                'Column': list(self.quality_metrics.missing_values.keys()),
                'Missing %': list(self.quality_metrics.missing_values.values())
            })
            if not missing_df.empty:
                missing_df = missing_df.sort_values('Missing %', ascending=True)
                ax.barh(missing_df['Column'], missing_df['Missing %'], color='orange')
                ax.set_title('Missing Values by Column')
                ax.set_xlabel('Missing %')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Missing Values')

        # 2. Outlier counts
        if self.quality_metrics.outlier_counts:
            ax = axes[0, 1]
            outlier_df = pd.DataFrame({
                'Column': list(self.quality_metrics.outlier_counts.keys()),
                'Outliers': list(self.quality_metrics.outlier_counts.values())
            })
            if not outlier_df.empty:
                outlier_df = outlier_df.sort_values('Outliers', ascending=True)
                ax.barh(outlier_df['Column'], outlier_df['Outliers'], color='red')
                ax.set_title('Outlier Counts by Column')
                ax.set_xlabel('Number of Outliers')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Outliers Found', ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Outliers')

        # 3. Correlation matrix heatmap
        if self.quality_metrics.correlation_matrix is not None:
            ax = axes[1, 0]
            corr = self.quality_metrics.correlation_matrix
            im = ax.imshow(corr, cmap='coolwarm', aspect='auto')
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(corr.columns, fontsize=8)
            ax.set_title('Correlation Matrix')
            plt.colorbar(im, ax=ax)
        else:
            axes[1, 0].text(0.5, 0.5, 'Not enough numerical columns\nfor correlation matrix',
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Correlation Matrix')

        # 4. Quality summary
        ax = axes[1, 1]
        quality_data = {
            'Missing Cols': len(self.quality_metrics.missing_values),
            'Outlier Cols': len(self.quality_metrics.outlier_counts),
            'Duplicate %': self.quality_metrics.duplicate_percentage,
            'Quality Score': self._calculate_quality_score()
        }
        bars = ax.bar(quality_data.keys(), quality_data.values(),
                     color=['orange', 'red', 'blue', 'green'])
        ax.set_title('Quality Summary')
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()