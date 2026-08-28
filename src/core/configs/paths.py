"""
Path Configuration for European Insurance Federated Learning
Centralized path management for all project directories
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ProjectPaths:
    """Project path configuration"""

    # Root directory (project root)
    root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.parent)

    # Data directories - only create when needed
    data_root: Optional[Path] = None
    data_processed: Optional[Path] = None

    # Output directories - only create when needed
    outputs_root: Optional[Path] = None
    reports_root: Optional[Path] = None
    reports_analysis: Optional[Path] = None

    # Logs
    logs_root: Optional[Path] = None

    def __post_init__(self):
        """Initialize all paths after dataclass creation"""
        # Only define paths, don't create directories
        self.data_root = self.root / "data"
        self.data_processed = self.data_root / "processed"

        self.outputs_root = self.root / "outputs"
        self.reports_root = self.outputs_root / "reports"
        self.reports_analysis = self.reports_root / "analysis"

        self.logs_root = self.root / "logs"

    def ensure_dirs(self, paths_to_create: list = None):
        """
        Create only specified directories

        Parameters:
        -----------
        paths_to_create : list
            List of directory paths to create
        """
        if paths_to_create is None:
            return

        for path in paths_to_create:
            if path is not None and not path.exists():
                path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, str]:
        """Convert paths to dictionary of strings"""
        return {
            'root': str(self.root),
            'data_processed': str(self.data_processed),
            'reports_analysis': str(self.reports_analysis),
            'logs_root': str(self.logs_root),
        }


# Singleton instance
_project_paths = None


def get_paths() -> ProjectPaths:
    """Get singleton ProjectPaths instance"""
    global _project_paths
    if _project_paths is None:
        _project_paths = ProjectPaths()
    return _project_paths


def ensure_directories(create_all: bool = True, specific: Optional[list] = None):
    """
    Ensure directories exist - only creates specified ones

    Parameters:
    -----------
    create_all : bool
        If True, creates only essential directories
    specific : Optional[list]
        List of specific directory keys to create
    """
    paths = get_paths()

    if specific is not None:
        dirs_to_create = []
        for key in specific:
            path = getattr(paths, key, None)
            if path is not None:
                dirs_to_create.append(path)
    else:
        # Only create essential directories
        dirs_to_create = [
            paths.data_processed,
            paths.reports_analysis,
            paths.logs_root,
        ]

    paths.ensure_dirs(dirs_to_create)


def get_dataset_path(dataset_name: str = "european_insurance_federated") -> Path:
    """Get path to a specific dataset"""
    paths = get_paths()
    return paths.data_processed / dataset_name


def get_report_path(report_type: str = "analysis") -> Path:
    """Get path for reports"""
    paths = get_paths()

    if report_type == "analysis":
        return paths.reports_analysis
    return paths.reports_root