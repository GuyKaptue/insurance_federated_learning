#!/usr/bin/env python3
"""
Dataset Generation Script
Generates federated and centralized insurance datasets
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.synthetic.data_generator import EuropeanInsuranceFederatedDataset
from src.core.configs.paths import get_paths, ensure_directories


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate European Insurance Federated Dataset"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of samples per client (default: 5000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        help="Dataset version (default: v1.0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/european_insurance_federated",
        help="Output directory (default: data/processed/european_insurance_federated)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "npy"],
        default="parquet",
        help="Save format: parquet or npy (default: parquet)"
    )
    parser.add_argument(
        "--no-centralized",
        action="store_true",
        help="Do not generate centralized dataset"
    )
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()

    # Set output directory
    output_dir = Path(args.output)

    print("=" * 80)
    print("🏦 European Insurance Federated Dataset Generator")
    print("=" * 80)
    print(f"📝 Samples per client: {args.samples:,}")
    print(f"🔢 Random seed: {args.seed}")
    print(f"📊 Version: {args.version}")
    print(f"📁 Output: {output_dir}")
    print(f"💾 Format: {args.format}")
    print(f"📦 Centralized: {'No' if args.no_centralized else 'Yes'}")
    print("=" * 80)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    generator = EuropeanInsuranceFederatedDataset(
        samples_per_client=args.samples,
        random_seed=args.seed,
        dataset_version=args.version
    )

    start_time = datetime.now()

    federated_data, centralized_data = generator.generate_dataset(
        output_dir=str(output_dir),
        save_format=args.format,
        generate_centralized=not args.no_centralized
    )

    end_time = datetime.now()
    generation_time = (end_time - start_time).total_seconds()

    print("\n" + "=" * 80)
    print("📊 Dataset Summary")
    print("=" * 80)
    print(f"✅ Generation completed in {generation_time:.2f} seconds")
    print(f"📁 Dataset location: {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())