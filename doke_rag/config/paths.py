"""
DOKE-RAG Path Configuration Module

This module provides centralized path management for the entire project.
It uses environment variables and relative paths to ensure portability.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ======================
# Project Root
# ======================

def get_project_root() -> Path:
    """
    Get the project root directory.

    Assumes this file is at: doke_rag/config/paths.py
    Project root is two levels up.
    """
    return Path(__file__).parent.parent.parent


PROJECT_ROOT = get_project_root()

# ======================
# Data Directories
# ======================

# Raw data storage
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXPERT_GRAPHS_DIR = DATA_DIR / "expert_graphs"
LLM_GRAPHS_DIR = DATA_DIR / "llm_graphs"

# Working directory (default)
DEFAULT_WORKING_DIR = PROJECT_ROOT / "data" / "run_data"

# ======================
# Evaluation Directories
# ======================

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
EVALUATION_RESULTS_DIR = EVALUATION_DIR / "results"
EVALUATION_OUTPUT_DIR = EVALUATION_DIR / "outputs"

# ======================
# Environment Variables
# ======================

def get_env_path(var_name: str, default: Path | None = None) -> Path:
    """
    Get a path from environment variable.

    Args:
        var_name: Environment variable name
        default: Default path if variable not set

    Returns:
        Path object
    """
    value = os.getenv(var_name)
    if value:
        return Path(value)
    return default or PROJECT_ROOT


# Working directory (can be overridden by env var)
WORKING_DIR = get_env_path("WORKING_DIR", default=DEFAULT_WORKING_DIR)

# Results directory
RESULTS_DIR = get_env_path("RESULTS_DIR", default=EVALUATION_RESULTS_DIR)

# Expert graphs directory
EXPERT_GRAPHS_PATH = get_env_path("EXPERT_GRAPHS_DIR", default=EXPERT_GRAPHS_DIR)

# ======================
# Config Files
# ======================

# .env file location
ENV_FILE = get_env_path("ENV_FILE", default=PROJECT_ROOT / ".env")

# Excel/CSV results
EXCEL_RESULTS = get_env_path("EXCEL_RESULTS", default=EVALUATION_DIR / "Consolidated_Report.xlsx")
CSV_CHAMPIONSHIP = get_env_path("CSV_CHAMPIONSHIP", default=EVALUATION_DIR / "Championship_Rankings_Final.csv")

# ======================
# Helper Functions
# ======================

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, create if it doesn't."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_path(path: str | Path) -> Path:
    """
    Resolve a path relative to project root.

    If path is absolute, return as-is.
    If path is relative, resolve from project root.

    Args:
        path: Path to resolve

    Returns:
        Resolved Path object
    """
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


# ======================
# Legacy Support
# ======================

# For backward compatibility with old scripts
# These map old absolute paths to new relative paths
PATH_MAPPINGS = {
    # Old StruMech project paths
    "D:\\Desktop\\StruMech GraphRAG\\StruMech": PROJECT_ROOT,
    "D:\\Desktop\\StruMech GraphRAG\\StruMech\\run_data": WORKING_DIR,
    "D:\\Desktop\\StruMech GraphRAG\\StruMech\\evaluation": EVALUATION_DIR,
    "D:\\Desktop\\StruMech GraphRAG\\StruMech\\QA": RESULTS_DIR,
    "D:\\Desktop\\StruMech GraphRAG\\StruMech\\.env": ENV_FILE,

    # Old gdf project paths
    "D:\\Desktop\\gdf\\graphRAG-camel\\LightRAG\\run_data": WORKING_DIR,
    "D:\\Desktop\\gdf\\graphRAG-camel\\LightRAG\\data": RAW_DATA_DIR,
    "D:\\Desktop\\gdf\\graphRAG-camel\\LightRAG\\QA": RESULTS_DIR,
}


def migrate_old_path(old_path: str) -> Path:
    """
    Migrate old absolute path to new relative path.

    Args:
        old_path: Old absolute path string

    Returns:
        New relative Path object
    """
    for old_prefix, new_path in PATH_MAPPINGS.items():
        if old_path.startswith(old_prefix):
            # Replace old prefix with new path
            relative_part = old_path[len(old_prefix):].lstrip("\\").strip()
            return new_path / relative_part

    # If no mapping found, return as-is (but warn)
    import warnings
    warnings.warn(f"No mapping found for path: {old_path}")
    return Path(old_path)
