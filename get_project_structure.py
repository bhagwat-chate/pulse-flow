# scripts/git_repo_tree.py
"""
================================================================================
 Git Repository Tree Visualizer
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : scripts.git_repo_tree
- Version     : 1.0.0
- Created on  : 2025-10-10
- Environment : Python 3.11.13 | Git CLI Required
================================================================================

Description
-----------
Utility script to generate and print the directory tree of all Git-tracked
files within the repository, excluding `__init__.py` and untracked files.

Architecture Context
--------------------
Layer:        Developer Utility
Usage:        For documentation, audits, and version-controlled module maps

Key Responsibilities
--------------------
• Use `git ls-files` to list tracked repository files.
• Build a nested tree-like dictionary representation of directories.
• Print a formatted ASCII tree view with hierarchy lines.

Engineering Standards
---------------------
• Cross-platform (works on Windows, Linux, macOS).
• Excludes unnecessary boilerplate like `__init__.py`.
• Handles empty repos gracefully.
• Uses behavior-oriented docstrings and modular structure.
"""

import os
import subprocess
from prod_assistant.core.globals import LOGGER


def get_git_tracked_files() -> list[str]:
    """
    Retrieve all Git-tracked files using `git ls-files`.

    Behavior
    --------
    - Executes a subprocess call to list all tracked files.
    - Splits output lines into a clean list.
    - Returns an empty list if the repository is not initialized.

    Returns
    -------
    list[str]
        List of file paths relative to the repository root.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--directory"],
            capture_output=True,
            text=True,
            check=True,
        )
        files = [f for f in result.stdout.splitlines() if not f.endswith("__init__.py")]
        # LOGGER.info("Fetched tracked files", count=len(files))
        return files
    except subprocess.CalledProcessError as e:
        # LOGGER.warning("Git repository not initialized or empty", error=str(e))
        return []


def build_file_tree(files: list[str]) -> dict:
    """
    Build a nested dictionary structure representing a file tree.

    Behavior
    --------
    - Splits file paths by directory separators.
    - Recursively constructs dictionary nodes for each subdirectory.
    - Returns an empty dictionary if no files are provided.

    Parameters
    ----------
    files : list[str]
        List of file paths relative to repo root.

    Returns
    -------
    dict
        Hierarchical dictionary representing directory structure.
    """
    tree = {}
    for file in files:
        parts = file.split("/")
        cur = tree
        for part in parts:
            cur = cur.setdefault(part, {})
    return tree


def print_tree(tree: dict, indent: str = ""):
    """
    Recursively print the directory tree in formatted ASCII style.

    Behavior
    --------
    - Uses visual connectors (`├──`, `└──`, `│`) to indicate hierarchy.
    - Adjusts indentation dynamically based on depth.
    - Handles both files and nested directories.

    Parameters
    ----------
    tree : dict
        Nested dictionary structure built by `build_file_tree`.
    indent : str, optional
        Current indentation level for nested printing.
    """
    for i, (name, children) in enumerate(sorted(tree.items())):
        connector = "└── " if i == len(tree) - 1 else "├── "
        print(indent + connector + name)
        if children:
            new_indent = indent + ("    " if i == len(tree) - 1 else "│   ")
            print_tree(children, new_indent)


def main():
    """
    Main execution entrypoint for the Git repository tree printer.

    Behavior
    --------
    - Retrieves all tracked files.
    - Builds directory hierarchy.
    - Prints the tree to the console.
    """
    files = get_git_tracked_files()
    if not files:
        print("No tracked files found or repository not initialized.")
        return
    tree = build_file_tree(files)
    print_tree(tree)


if __name__ == "__main__":
    main()
