#!/bin/bash
# ============================================================
# European Insurance Federated Learning - Dataset Generator
# Generates, analyzes, and validates the complete dataset
# PyTorch Version - Python 3.11 Compatible
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

# Check Python version
check_python() {
    print_message "Checking Python version..."

    if command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
        PYTHON_VERSION=$(python3.11 --version 2>&1 | awk '{print $2}')
        print_success "Found Python $PYTHON_VERSION"
        export PYTHON_CMD
        return 0
    fi

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        if [[ $PYTHON_VERSION == 3.11* ]] || [[ $PYTHON_VERSION == 3.12* ]]; then
            PYTHON_CMD="python3"
            print_success "Found Python $PYTHON_VERSION"
            export PYTHON_CMD
            return 0
        fi
    fi

    print_error "Python 3.11+ not found. Please install Python 3.11."
    exit 1
}

# Setup virtual environment
setup_venv() {
    print_message "Setting up virtual environment..."

    if [ -d "venv" ]; then
        print_message "Removing existing virtual environment..."
        rm -rf venv
    fi

    print_message "Creating virtual environment with Python $($PYTHON_CMD --version)..."
    $PYTHON_CMD -m venv venv

    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Install dependencies
install_dependencies() {
    print_message "Installing dependencies..."

    pip install --upgrade pip

    print_message "Installing core dependencies..."
    pip install numpy pandas scipy pyarrow scikit-learn

    print_message "Installing ML libraries..."
    pip install xgboost lightgbm

    print_message "Installing PyTorch..."
    pip install torch torchvision

    print_message "Installing Flower Federated Learning..."
    pip install flwr

    print_message "Installing additional dependencies..."
    pip install fastapi uvicorn pydantic mlflow matplotlib seaborn plotly python-dotenv click loguru tqdm rich pyyaml joblib

    print_success "All dependencies installed"
}

# Generate dataset (creates only needed directories)
generate_dataset() {
    local samples=$1
    local output_dir=$2

    print_header "Step 1: Generating Dataset"
    print_message "Samples per client: $samples"
    print_message "Output directory: $output_dir"

    # The Python script will create only the directories it needs
    $PYTHON_CMD scripts/generate_dataset.py \
        --samples "$samples" \
        --output "$output_dir"

    if [ $? -eq 0 ]; then
        print_success "Dataset generated successfully"
    else
        print_error "Dataset generation failed"
        exit 1
    fi
}

# Analyze dataset (creates only needed directories)
analyze_dataset() {
    local data_path=$1
    local output_dir=$2

    print_header "Step 2: Analyzing Dataset"
    print_message "Data path: $data_path"
    print_message "Output directory: $output_dir"

    # The Python script will create only the directories it needs
    $PYTHON_CMD scripts/analyze_data.py \
        --data-path "$data_path" \
        --output "$output_dir" \
        --plot

    if [ $? -eq 0 ]; then
        print_success "Dataset analysis completed"
    else
        print_error "Dataset analysis failed"
        exit 1
    fi
}

# Show summary
show_summary() {
    local data_path=$1
    local reports_dir=$2
    local samples=$3

    print_header "✅ Dataset Generation Complete!"

    echo ""
    echo "📊 Dataset Summary:"
    echo "  - Location: $data_path"
    echo "  - Reports: $reports_dir"
    echo "  - Samples per client: $samples"
    echo "  - Total clients: 4 (Germany, France, Belgium, Switzerland)"
    echo "  - Total samples: $((samples * 4))"
    echo ""
    echo "📁 Output Structure (only created directories):"

    # Check what was actually created
    if [ -d "$data_path/federated" ]; then
        echo "  $data_path/"
        echo "  ├── federated/"
        for client in $(ls "$data_path/federated" 2>/dev/null); do
            echo "  │   ├── $client/"
            echo "  │   │   ├── train.parquet"
            echo "  │   │   ├── val.parquet"
            echo "  │   │   └── test.parquet"
        done
        if [ -d "$data_path/centralized" ]; then
            echo "  ├── centralized/"
            echo "  │   ├── train.parquet"
            echo "  │   ├── val.parquet"
            echo "  │   └── test.parquet"
        fi
        if [ -f "$data_path/metadata.json" ]; then
            echo "  ├── metadata.json"
        fi
        if [ -f "$data_path/dataset_info.json" ]; then
            echo "  └── dataset_info.json"
        fi
    fi

    if [ -d "$reports_dir/analysis" ]; then
        echo ""
        echo "📊 Reports:"
        echo "  $reports_dir/"
        if [ -d "$reports_dir/analysis" ]; then
            echo "  └── analysis/         # Analysis reports"
        fi
    fi
    echo ""
}

# Show usage
show_usage() {
    cat << EOF
Usage: ./scripts/generator.sh [OPTIONS]

Description:
    Generate the complete European Insurance Federated Learning dataset.
    Only creates directories that are actually used.

Options:
    -s, --samples N     Number of samples per client (default: 5000)
    -h, --help          Show this help message

Examples:
    ./scripts/generator.sh
    ./scripts/generator.sh --samples 10000

EOF
}

# Main execution
main() {
    SAMPLES=5000
    OUTPUT_DIR="data/processed/european_insurance_federated"
    REPORTS_DIR="outputs/reports"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -s|--samples)
                SAMPLES="$2"
                shift 2
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    print_header "🏦 European Insurance Federated Dataset Generator"

    echo "Configuration:"
    echo "  - Samples per client: $SAMPLES"
    echo "  - Total clients: 4"
    echo "  - Total samples: $((SAMPLES * 4))"
    echo "  - Output directory: $OUTPUT_DIR"
    echo "  - Framework: PyTorch"
    echo ""

    check_python
    setup_venv
    install_dependencies

    # Only create the main data directory (the rest will be created by scripts)
    mkdir -p data

    generate_dataset "$SAMPLES" "$OUTPUT_DIR"
    analyze_dataset "$OUTPUT_DIR" "$REPORTS_DIR"/analysis
    show_summary "$OUTPUT_DIR" "$REPORTS_DIR" "$SAMPLES"

    print_success "🎉 Dataset generation complete!"
}

main "$@"