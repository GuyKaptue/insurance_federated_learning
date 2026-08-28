#!/usr/bin/env python3
"""
Data Analysis Script - Professional Implementation
Analyzes federated and centralized datasets with deterministic outputs
Runs all clients and centralized dataset by default
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path - FIXED for scripts/ folder
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.configs.paths import (
    ensure_directories,
    get_dataset_path,
    get_paths,
    get_report_path
)
from src.core.loader.data_loader import DataLoader, DatasetSplit
from src.core.loader.data_preprocessor import DataPreprocessor
from src.core.loader.data_quality_analyzer import DataQualityAnalyzer, QualityMetrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze European Insurance Federated Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all clients and centralized (default)
  python analyze_data.py

  # Analyze specific client only
  python analyze_data.py --client germany_allianz

  # Skip centralized analysis
  python analyze_data.py --no-centralized

  # Analyze with plots
  python analyze_data.py --plot

  # Analyze with custom paths
  python analyze_data.py --data-path data/my_dataset --output reports/my_analysis
        """
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset (default: data/processed/european_insurance_federated)"
    )
    parser.add_argument(
        "--client",
        type=str,
        help="Analyze specific client only (default: all clients)"
    )
    parser.add_argument(
        "--no-centralized",
        action="store_true",
        help="Skip centralized dataset analysis (default: analyze centralized)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for reports (default: outputs/reports/analysis)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots"
    )
    parser.add_argument(
        "--no-create-dirs",
        action="store_true",
        help="Do not create directories"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    return parser.parse_args()


def setup_output_directory(output_path: Path, client_id: Optional[str] = None) -> Path:
    """
    Setup deterministic output directory structure

    Parameters:
    -----------
    output_path : Path
        Base output path
    client_id : Optional[str]
        Client ID for client-specific subdirectory (None for centralized)

    Returns:
    --------
    Path
        Final output directory path
    """
    if client_id is not None:
        output_dir = output_path / "clients" / client_id
    else:
        # Centralized reports go directly under the output path
        output_dir = output_path / "centralized"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_deterministic_filename(base_name: str, extension: str, suffix: Optional[str] = None) -> str:
    """
    Generate deterministic filename without timestamps

    Parameters:
    -----------
    base_name : str
        Base filename
    extension : str
        File extension (without dot)
    suffix : Optional[str]
        Optional suffix (e.g., 'summary', 'detailed')

    Returns:
    --------
    str
        Deterministic filename
    """
    if suffix:
        return f"{base_name}_{suffix}.{extension}"
    return f"{base_name}.{extension}"


def analyze_client_data(
    client_id: str,
    data: DatasetSplit,
    analyzer: DataQualityAnalyzer,
    output_path: Path,
    plot: bool = False,
    verbose: bool = False
) -> Tuple[Dict[str, Any], Path]:
    """
    Analyze data for a single client

    Parameters:
    -----------
    client_id : str
        Client identifier
    data : DatasetSplit
        Client data splits
    analyzer : DataQualityAnalyzer
        Data quality analyzer instance
    output_path : Path
        Output directory path
    plot : bool
        Whether to generate plots
    verbose : bool
        Verbose output

    Returns:
    --------
    Tuple[Dict, Path]
        (Report dictionary, output directory path)
    """
    # Setup output directory for client
    client_output_dir = setup_output_directory(output_path, client_id)

    # Prepare client data
    client_data = {
        'train': data.train,
        'val': data.val,
        'test': data.test
    }

    # Generate report
    report = analyzer.generate_client_report(
        client_data,
        client_id,
        output_dir=str(client_output_dir)
    )

    # Save client report with deterministic filename
    report_file = client_output_dir / get_deterministic_filename(
        f"quality_report_{client_id}", "json"
    )
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    if verbose:
        print(f"  📄 Report saved: {report_file}")

    # Generate plots if requested
    if plot:
        plot_dir = client_output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Generate distribution plots
            generate_client_plots(
                client_data,
                client_id,
                plot_dir
            )
            if verbose:
                print(f"  📊 Plots saved: {plot_dir}")
        except Exception as e:
            print(f"  ⚠️ Could not generate plots: {e}")

    return report, client_output_dir


def generate_client_plots(
    client_data: Dict[str, pd.DataFrame],
    client_id: str,
    output_dir: Path,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Generate plots for client data analysis

    Parameters:
    -----------
    client_data : Dict[str, pd.DataFrame]
        Dictionary containing 'train', 'val', 'test' dataframes
    client_id : str
        Client identifier
    output_dir : Path
        Output directory for plots
    figsize : Tuple[int, int]
        Figure size
    """
    # Combine all splits for analysis
    combined_df = pd.concat([
        client_data.get('train', pd.DataFrame()),
        client_data.get('val', pd.DataFrame()),
        client_data.get('test', pd.DataFrame())
    ], ignore_index=True)

    if len(combined_df) == 0:
        print(f"⚠️ No data available for client {client_id}")
        return

    # Get numerical columns
    num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()

    # 1. Feature distributions (first 12 columns)
    if num_cols:
        n_cols = min(12, len(num_cols))
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(num_cols[:n_cols]):
            ax = axes[i]
            combined_df[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col[:20]}', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Frequency')

        for j in range(i + 1, 12):
            axes[j].set_visible(False)

        plt.suptitle(f'Feature Distributions - {client_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / get_deterministic_filename(
            f"distributions_{client_id}", "png"
        ), dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Correlation matrix
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = combined_df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            cmap='coolwarm',
            center=0,
            annot=False,
            ax=ax,
            square=True,
            linewidths=0.5
        )
        ax.set_title(f'Correlation Matrix - {client_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / get_deterministic_filename(
            f"correlation_{client_id}", "png"
        ), dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Missing values
    missing_data = combined_df.isnull().sum()
    missing_data = missing_data[missing_data > 0]

    if not missing_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        missing_data.sort_values().plot(kind='barh', ax=ax, color='orange')
        ax.set_title(f'Missing Values - {client_id}', fontsize=14)
        ax.set_xlabel('Number of Missing Values')
        ax.set_ylabel('Columns')
        plt.tight_layout()
        plt.savefig(output_dir / get_deterministic_filename(
            f"missing_values_{client_id}", "png"
        ), dpi=150, bbox_inches='tight')
        plt.close()

    # 4. Target distributions
    target_cols = [
        'claim_risk_score', 'customer_lifetime_value',
        'will_renew', 'is_high_risk', 'customer_segment'
    ]
    target_cols = [col for col in target_cols if col in combined_df.columns]

    if target_cols:
        n_targets = len(target_cols)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, col in enumerate(target_cols):
            ax = axes[i]
            if combined_df[col].dtype in ['float64', 'int64'] and combined_df[col].nunique() > 20:
                combined_df[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
                ax.set_title(f'{col}', fontsize=11)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
            else:
                value_counts = combined_df[col].value_counts()
                value_counts.plot(kind='bar', ax=ax, alpha=0.7)
                ax.set_title(f'{col}', fontsize=11)
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)

        for j in range(i + 1, 6):
            axes[j].set_visible(False)

        plt.suptitle(f'Target Distributions - {client_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / get_deterministic_filename(
            f"targets_{client_id}", "png"
        ), dpi=150, bbox_inches='tight')
        plt.close()


def analyze_centralized_data(
    centralized_data: DatasetSplit,
    analyzer: DataQualityAnalyzer,
    output_path: Path,
    plot: bool = False,
    verbose: bool = False
) -> Tuple[Dict[str, Any], Path]:
    """
    Analyze centralized dataset

    Parameters:
    -----------
    centralized_data : DatasetSplit
        Centralized dataset splits
    analyzer : DataQualityAnalyzer
        Data quality analyzer instance
    output_path : Path
        Output directory path (centralized reports go directly here)
    plot : bool
        Whether to generate plots
    verbose : bool
        Verbose output

    Returns:
    --------
    Tuple[Dict, Path]
        (Report dictionary, output directory path)
    """
    # Setup output directory for centralized (NOT under clients/)
    central_output_dir = setup_output_directory(output_path, None)

    # Combine all splits
    combined_df = pd.concat([
        centralized_data.train,
        centralized_data.val,
        centralized_data.test
    ], ignore_index=True)

    # Analyze
    report = analyzer.analyze(combined_df)

    # Convert to dict for JSON serialization
    report_dict = {
        'total_rows': report.total_rows,
        'total_columns': report.total_columns,
        'missing_values': report.missing_values,
        'duplicate_rows': report.duplicate_rows,
        'duplicate_percentage': report.duplicate_percentage,
        'outlier_counts': report.outlier_counts,
        'outlier_percentages': report.outlier_percentages,
        'unique_values': report.unique_values,
        'value_counts': report.value_counts,
        'column_types': report.column_types,
        'summary_stats': report.summary_stats
    }

    # Save centralized report
    report_file = central_output_dir / get_deterministic_filename(
        "quality_report_centralized", "json"
    )
    with open(report_file, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)

    if verbose:
        print(f"  📄 Report saved: {report_file}")

    # Generate quality summary report
    quality_summary = {
        'data_quality_summary': {
            'total_samples': report.total_rows,
            'total_features': report.total_columns,
            'missing_value_count': len(report.missing_values),
            'duplicate_percentage': report.duplicate_percentage,
            'overall_quality_score': analyzer._calculate_quality_score()
        },
        'recommendations': analyzer._get_recommendations()
    }

    # Save quality summary
    summary_file = central_output_dir / get_deterministic_filename(
        "quality_summary_centralized", "json"
    )
    with open(summary_file, 'w') as f:
        json.dump(quality_summary, f, indent=2, default=str)

    # Generate plots if requested
    if plot:
        plot_dir = central_output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        try:
            generate_centralized_plots(
                centralized_data,
                plot_dir
            )
            if verbose:
                print(f"  📊 Plots saved: {plot_dir}")
        except Exception as e:
            print(f"  ⚠️ Could not generate plots: {e}")

    return quality_summary, central_output_dir


def generate_centralized_plots(
    centralized_data: DatasetSplit,
    output_dir: Path,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Generate plots for centralized dataset analysis

    Parameters:
    -----------
    centralized_data : DatasetSplit
        Centralized dataset splits
    output_dir : Path
        Output directory for plots
    figsize : Tuple[int, int]
        Figure size
    """
    # Combine all splits
    combined_df = pd.concat([
        centralized_data.train,
        centralized_data.val,
        centralized_data.test
    ], ignore_index=True)

    # 1. Client distribution
    if 'client_id' in combined_df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        client_counts = combined_df['client_id'].value_counts()
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = client_counts.plot(kind='bar', ax=ax, color=colors[:len(client_counts)])
        ax.set_title('Client Distribution in Centralized Dataset', fontsize=14)
        ax.set_xlabel('Client')
        ax.set_ylabel('Number of Samples')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(output_dir / get_deterministic_filename(
            "centralized_client_distribution", "png"
        ), dpi=150, bbox_inches='tight')
        plt.close()

    # 2. Target distributions
    target_cols = [
        'claim_risk_score', 'customer_lifetime_value',
        'will_renew', 'is_high_risk', 'customer_segment'
    ]
    target_cols = [col for col in target_cols if col in combined_df.columns]

    if target_cols:
        n_targets = len(target_cols)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(target_cols):
            ax = axes[i]
            if combined_df[col].dtype in ['float64', 'int64'] and combined_df[col].nunique() > 20:
                combined_df[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
                ax.set_title(f'{col}', fontsize=11)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
            else:
                value_counts = combined_df[col].value_counts()
                value_counts.plot(kind='bar', ax=ax, alpha=0.7)
                ax.set_title(f'{col}', fontsize=11)
                ax.set_xlabel('Class')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)

        for j in range(i + 1, 6):
            axes[j].set_visible(False)

        plt.suptitle('Target Distributions - Centralized Dataset', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / get_deterministic_filename(
            "centralized_targets", "png"
        ), dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Feature distributions (top 12)
    num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        n_cols = min(12, len(num_cols))
        fig, axes = plt.subplots(3, 4, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(num_cols[:n_cols]):
            ax = axes[i]
            combined_df[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'{col[:20]}', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('Frequency')

        for j in range(i + 1, 12):
            axes[j].set_visible(False)

        plt.suptitle('Feature Distributions - Centralized Dataset', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / get_deterministic_filename(
            "centralized_distributions", "png"
        ), dpi=150, bbox_inches='tight')
        plt.close()


def generate_summary_report(
    all_reports: Dict[str, Any],
    output_dir: Path,
    verbose: bool = False
) -> Path:
    """
    Generate comprehensive summary report

    Parameters:
    -----------
    all_reports : Dict[str, Any]
        Dictionary of all reports
    output_dir : Path
        Output directory
    verbose : bool
        Verbose output

    Returns:
    --------
    Path
        Path to summary report
    """
    summary = {
        'analysis_date': str(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')),
        'clients_analyzed': len([k for k in all_reports.keys() if k != 'centralized']),
        'centralized_available': 'centralized' in all_reports,
        'client_summaries': {}
    }

    # Client summaries
    for client_id, report in all_reports.items():
        if client_id == 'centralized':
            continue

        summary['client_summaries'][client_id] = {
            'quality_score': report['data_quality_summary']['overall_quality_score'],
            'total_samples': report['data_quality_summary']['total_samples'],
            'features': report['data_quality_summary']['total_features'],
            'missing_values': report['data_quality_summary']['missing_value_count'],
            'duplicates': report['data_quality_summary']['duplicate_percentage'],
            'recommendations': report['recommendations'][:5]
        }

    # Centralized summary
    if 'centralized' in all_reports:
        report = all_reports['centralized']
        summary['centralized'] = {
            'quality_score': report['data_quality_summary']['overall_quality_score'],
            'total_samples': report['data_quality_summary']['total_samples'],
            'features': report['data_quality_summary']['total_features'],
            'missing_values': report['data_quality_summary']['missing_value_count'],
            'duplicates': report['data_quality_summary']['duplicate_percentage'],
            'recommendations': report['recommendations'][:5]
        }

    # Save summary as JSON
    summary_json_path = output_dir / get_deterministic_filename("summary_report", "json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    if verbose:
        print(f"  📄 Summary JSON: {summary_json_path}")

    # Create markdown report
    summary_md_path = generate_markdown_report(summary, output_dir)

    if verbose:
        print(f"  📄 Summary Markdown: {summary_md_path}")

    return summary_json_path


def generate_markdown_report(summary: Dict[str, Any], output_dir: Path) -> Path:
    """
    Generate markdown summary report

    Parameters:
    -----------
    summary : Dict[str, Any]
        Summary dictionary
    output_dir : Path
        Output directory

    Returns:
    --------
    Path
        Path to markdown report
    """
    md_path = output_dir / get_deterministic_filename("summary_report", "md")

    with open(md_path, 'w') as f:
        f.write("# 📊 Dataset Analysis Summary Report\n\n")
        f.write(f"**Analysis Date:** {summary['analysis_date']}\n\n")

        f.write("---\n\n")

        f.write("## 📈 Overview\n\n")
        f.write(f"- **Clients Analyzed:** {summary['clients_analyzed']}\n")
        f.write(f"- **Centralized Dataset:** {'Yes' if summary['centralized_available'] else 'No'}\n\n")

        f.write("---\n\n")

        f.write("## 🏦 Client Summaries\n\n")

        for client_id, stats in summary.get('client_summaries', {}).items():
            f.write(f"### {client_id}\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Quality Score | {stats['quality_score']:.1f}/100 |\n")
            f.write(f"| Total Samples | {stats['total_samples']:,} |\n")
            f.write(f"| Features | {stats['features']} |\n")
            f.write(f"| Missing Values | {stats['missing_values']} columns |\n")
            f.write(f"| Duplicates | {stats['duplicates']:.2f}% |\n\n")

            if stats.get('recommendations'):
                f.write("**Recommendations:**\n\n")
                for rec in stats['recommendations'][:3]:
                    f.write(f"- {rec}\n")
                f.write("\n")

        f.write("---\n\n")

        if summary.get('centralized'):
            f.write("## 📦 Centralized Dataset\n\n")
            stats = summary['centralized']
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Quality Score | {stats['quality_score']:.1f}/100 |\n")
            f.write(f"| Total Samples | {stats['total_samples']:,} |\n")
            f.write(f"| Features | {stats['features']} |\n")
            f.write(f"| Missing Values | {stats['missing_values']} columns |\n")
            f.write(f"| Duplicates | {stats['duplicates']:.2f}% |\n\n")

            if stats.get('recommendations'):
                f.write("**Recommendations:**\n\n")
                for rec in stats['recommendations'][:3]:
                    f.write(f"- {rec}\n")
                f.write("\n")

        f.write("---\n\n")
        f.write("_Report generated by Insurance Federated Learning Data Quality Analyzer_\n")
        f.write("_All filenames are deterministic and reproducible_\n")

    return md_path


def main() -> int:
    """Main execution function"""
    args = parse_args()

    # Get paths
    paths = get_paths()

    # Ensure directories exist if not disabled
    if not args.no_create_dirs:
        ensure_directories(create_all=True)

    # Set data path
    if args.data_path is None:
        data_path = get_dataset_path("european_insurance_federated")
    else:
        data_path = Path(args.data_path)

    # Set output path
    if args.output is None:
        output_path = get_report_path("analysis")
    else:
        output_path = Path(args.output)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔍 European Insurance Dataset Analyzer")
    print("=" * 80)
    print(f"📁 Data path: {data_path}")
    print(f"📊 Output: {output_path}")
    print(f"🎯 Analysis mode: {'Specific Client' if args.client else 'All Clients + Centralized'}")
    print(f"🎨 Generate plots: {'Yes' if args.plot else 'No'}")
    print("=" * 80)

    # Check if data exists
    if not data_path.exists():
        print(f"❌ Data path does not exist: {data_path}")
        print("Run generate_dataset.py first or specify correct path")
        return 1

    # Load data
    loader = DataLoader(data_path)

    # Always load both federated and centralized data
    federated_data, centralized_data = loader.load_all(load_centralized=True)

    if not federated_data:
        print("❌ No federated data found to analyze")
        return 1

    # Determine clients to analyze
    if args.client:
        clients_to_analyze = [args.client]
        print(f"\n🎯 Analyzing specific client: {args.client}")
    else:
        clients_to_analyze = list(federated_data.keys())
        print(f"\n🎯 Analyzing all {len(clients_to_analyze)} clients")

    # Initialize analyzer
    analyzer = DataQualityAnalyzer()
    all_reports = {}

    # Analyze federated data
    print("\n📊 Analyzing federated data...")
    print("-" * 40)

    for idx, client_id in enumerate(clients_to_analyze, 1):
        if client_id not in federated_data:
            print(f"⚠️ Client {client_id} not found")
            continue

        print(f"\n[{idx}/{len(clients_to_analyze)}] 📊 Analyzing client: {client_id}")
        data = federated_data[client_id]

        report, _ = analyze_client_data(
            client_id=client_id,
            data=data,
            analyzer=analyzer,
            output_path=output_path,
            plot=args.plot,
            verbose=args.verbose
        )
        all_reports[client_id] = report

        # Print summary
        print(f"  ✅ Quality Score: {report['data_quality_summary']['overall_quality_score']:.1f}/100")
        print(f"  📊 Total Samples: {report['data_quality_summary']['total_samples']:,}")
        print(f"  🔢 Features: {report['data_quality_summary']['total_features']}")
        print(f"  ❗ Missing Values: {report['data_quality_summary']['missing_value_count']} columns")
        print(f"  🔄 Duplicates: {report['data_quality_summary']['duplicate_percentage']:.2f}%")

    # Analyze centralized data (always unless --no-centralized)
    if not args.no_centralized and centralized_data is not None:
        print(f"\n📊 Analyzing Centralized Dataset...")
        print("-" * 40)

        report, _ = analyze_centralized_data(
            centralized_data=centralized_data,
            analyzer=analyzer,
            output_path=output_path,
            plot=args.plot,
            verbose=args.verbose
        )

        all_reports['centralized'] = report

        print(f"  ✅ Quality Score: {report['data_quality_summary']['overall_quality_score']:.1f}/100")
        print(f"  📊 Total Samples: {report['data_quality_summary']['total_samples']:,}")
        print(f"  🔢 Features: {report['data_quality_summary']['total_features']}")
        print(f"  ❗ Missing Values: {report['data_quality_summary']['missing_value_count']} columns")
        print(f"  🔄 Duplicates: {report['data_quality_summary']['duplicate_percentage']:.2f}%")

    elif args.no_centralized:
        print(f"\n⏭️ Skipping centralized dataset analysis (--no-centralized)")

    # Generate summary report
    print("\n📝 Generating summary report...")
    print("-" * 40)
    summary_path = generate_summary_report(all_reports, output_path, verbose=args.verbose)

    print(f"\n✅ Analysis complete!")
    print(f"📁 Reports saved to: {output_path}/")
    print(f"📄 Summary report: {summary_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("📊 Analysis Summary")
    print("=" * 80)
    print(f"  - Clients analyzed: {len([k for k in all_reports.keys() if k != 'centralized'])}")
    print(f"  - Centralized analyzed: {'Yes' if 'centralized' in all_reports else 'No'}")
    print(f"  - Plots generated: {'Yes' if args.plot else 'No'}")
    print(f"  - Reports directory: {output_path}")
    print("=" * 80)

    # Show output structure
    print("\n📁 Output structure:")
    print(f"  {output_path}/")
    print(f"  ├── clients/")
    for client_id in all_reports.keys():
        if client_id != 'centralized':
            print(f"  │   ├── {client_id}/")
            print(f"  │   │   ├── quality_report_{client_id}.json")
            if args.plot:
                print(f"  │   │   └── plots/")
                print(f"  │   │       ├── distributions_{client_id}.png")
                print(f"  │   │       ├── correlation_{client_id}.png")
                print(f"  │   │       └── targets_{client_id}.png")
    if 'centralized' in all_reports:
        print(f"  ├── centralized/")
        print(f"  │   ├── quality_report_centralized.json")
        print(f"  │   ├── quality_summary_centralized.json")
        if args.plot:
            print(f"  │   └── plots/")
            print(f"  │       ├── centralized_client_distribution.png")
            print(f"  │       ├── centralized_targets.png")
            print(f"  │       └── centralized_distributions.png")
    print(f"  ├── summary_report.json")
    print(f"  └── summary_report.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())