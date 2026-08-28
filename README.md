# 🏦 Insurance Federated Learning Platform

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow Federated](https://img.shields.io/badge/TFF-0.52+-orange.svg)](https://www.tensorflow.org/federated)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/GuyKaptue/insurance-federated-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/GuyKaptue/insurance-federated-learning/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GuyKaptue/insurance-federated-learning/branch/main/graph/badge.svg)](https://codecov.io/gh/GuyKaptue/insurance-federated-learning)
[![GitHub release](https://img.shields.io/github/release/GuyKaptue/insurance-federated-learning.svg)](https://GitHub.com/GuyKaptue/insurance-federated-learning/releases/)
[![GitHub last commit](https://img.shields.io/github/last-commit/GuyKaptue/insurance-federated-learning.svg)](https://github.com/GuyKaptue/insurance-federated-learning/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/GuyKaptue/insurance-federated-learning.svg)](https://github.com/GuyKaptue/insurance-federated-learning/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/GuyKaptue/insurance-federated-learning.svg)](https://github.com/GuyKaptue/insurance-federated-learning/network)
[![GitHub issues](https://img.shields.io/github/issues/GuyKaptue/insurance-federated-learning.svg)](https://github.com/GuyKaptue/insurance-federated-learning/issues)

[![PyPI version](https://badge.fury.io/py/insurance-federated-learning.svg)](https://badge.fury.io/py/insurance-federated-learning)
[![Downloads](https://pepy.tech/badge/insurance-federated-learning)](https://pepy.tech/project/insurance-federated-learning)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://insurance-federated-learning.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXX)

## 📌 Overview

A production-ready federated learning platform for insurance analytics across 4 European countries:
- 🇩🇪 **Germany** (Allianz)
- 🇫🇷 **France** (AXA)
- 🇧🇪 **Belgium** (Ageas)
- 🇨🇭 **Switzerland** (Helvetia)

## ✨ Features

- 🏦 **Realistic Insurance Data**: Synthetic data with real-world statistics
- 🤖 **5 ML Tasks**: 2 Regression, 2 Binary Classification, 1 Multi-class
- 🔒 **Privacy-Preserving**: Federated learning with differential privacy
- 🚀 **Production Ready**: Docker, Kubernetes, FastAPI deployment
- 📊 **Comprehensive Monitoring**: Prometheus, Grafana dashboards

## 📦 Installation

### From PyPI
```bash
pip install insurance-federated-learning
```

### From Source
```bash
git clone https://github.com/GuyKaptue/insurance-federated-learning.git
cd insurance-federated-learning
pip install -e ".[dev]"
```

## 🚀 Quick Start

### Generate Dataset
```bash
insurance-fl-generate --samples 5000 --output data/processed
```

### Train Federated Model
```bash
insurance-fl-train --config config/federated_config.yaml
```

### Start API Server
```bash
insurance-fl-serve --host 0.0.0.0 --port 8000
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/usage.md)
- [API Reference](docs/api.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test category
pytest tests/unit/ -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Guy M. Kaptue T**

- GitHub: [@GuyKaptue](https://github.com/GuyKaptue)
- Email: guykaptue24@gmail.com
- LinkedIn: [Guy Kaptue](https://linkedin.com/in/guy-kaptue)

## 🙏 Acknowledgments

- OECD Statistics
- Eurostat Database
- European Insurance Consortium
- TensorFlow Federated Team
- Open Source Community

## 🔗 Links

- [GitHub Repository](https://github.com/GuyKaptue/insurance-federated-learning)
- [Issue Tracker](https://github.com/GuyKaptue/insurance-federated-learning/issues)
- [Documentation](https://insurance-federated-learning.readthedocs.io)
- [PyPI Package](https://pypi.org/project/insurance-federated-learning/)

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{kaptue_2024_insurance_fl,
  author = {Kaptue T, Guy M.},
  title = {Insurance Federated Learning: A Platform for European Insurance Analytics},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/GuyKaptue/insurance-federated-learning}
}
```

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GuyKaptue/insurance-federated-learning&type=Date)](https://star-history.com/#GuyKaptue/insurance-federated-learning&Date)
```

## Updated setup.py

```python
"""
Setup script for Insurance Federated Learning Project
"""

from setuptools import setup, find_packages
import os

# Read requirements
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='insurance-federated-learning',
    version='1.0.0',
    author='Guy M. Kaptue T',
    author_email='guykaptue24@gmail.com',
    description='Federated Learning Platform for European Insurance Analytics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GuyKaptue/insurance-federated-learning',
    project_urls={
        'Bug Tracker': 'https://github.com/GuyKaptue/insurance-federated-learning/issues',
        'Documentation': 'https://insurance-federated-learning.readthedocs.io',
        'Source Code': 'https://github.com/GuyKaptue/insurance-federated-learning',
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11,<3.13',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.2.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'isort>=5.12.0',
            'pre-commit>=3.0.0',
            'jupyter>=1.0.0',
        ],
        'docs': [
            'sphinx>=5.3.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=1.0.0',
            'mkdocs>=1.4.0',
            'mkdocs-material>=9.0.0',
        ],
        'cloud': [
            'boto3>=1.26.0',
            'google-cloud-storage>=2.8.0',
            'azure-storage-blob>=12.14.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'insurance-fl-generate=src.scripts.generate_dataset:main',
            'insurance-fl-train=src.scripts.run_federated_training:main',
            'insurance-fl-evaluate=src.scripts.evaluate_models:main',
            'insurance-fl-serve=src.scripts.deploy_api:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
```

## GitHub Repository Setup Commands

```bash
# Initialize Git repository
cd insurance_federated_learning
git init

# Add remote repository
git remote add origin https://github.com/GuyKaptue/insurance-federated-learning.git

# Create and switch to main branch
git branch -M main

# Add all files
git add .
git commit -m "Initial commit: European Insurance Federated Learning Platform"

# Push to GitHub
git push -u origin main

# Create and push tags
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Create development branch
git checkout -b develop
git push -u origin develop
