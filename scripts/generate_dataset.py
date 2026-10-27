#!/usr/bin/env python3
"""Dataset Generation Script
Generates federated and centralized insurance datasets
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from insurance_fl.data.generator import EuropeanInsuranceFederatedDataset

def main():
    import argparse as _argparse
    parser = _argparse.ArgumentParser(prog="insurance_fl.data.generator")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v1.0")
    parser.add_argument("--output", type=str, default="data/processed/european_insurance_federated")
    parser.add_ar    parser.add_ar    parser.add_ar    parser.add_ar    parser.add_ar    parser.aparser.add_argument("--no-quality-issues", action="store_true",
                        help="disable missing/outlier injection (baseline)")
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 80)
    print("European Insurance Federated Dataset Generator")
    print("=" * 80)
    print(f"Samples per client: {args.samples:,}")
    print(f"Random seed: {args.seed}")
    print(f"Version: {args.version}")
    print(f"Output: {output_dir}")
    print(f"Format: {args.format}")
    print(f"Quality issues: {'No' if args.no_quality_issues else 'Yes'}")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    gen = EuropeanInsuranceFederatedDataset(
        samples_per_client=args.samples,
        random_seed=args.seed,
        dataset_version=args.version,
        inject_quality_issues=not args.no_quality_issues,
    )

    start_time = datetime.now()

    gen.generate_dataset(output_dir=str(output_dir))

    end_time = datetime.now()
    generation_time = (end_time - start_time).total_seconds()

    print("
" + "=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"Generation completed in {generation_time:.2f} seconds")
    print(f"Dataset location: {output_dir}/")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
