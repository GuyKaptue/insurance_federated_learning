"""
src/core/configs/__init__.py
Configs Module - Centralized Configuration
"""

from src.core.configs.paths import (
    ProjectPaths,
    ensure_directories,
    get_dataset_path,
    get_paths,
    get_report_path,
)

__all__ = [
    'ProjectPaths',
    'get_paths',
    'ensure_directories',
    'get_dataset_path',
    'get_report_path'
]