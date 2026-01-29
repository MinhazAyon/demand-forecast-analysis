# core/utils/io.py

# --- Standard Python imports ---
import os
from functools import lru_cache
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, Tuple, Union

# --- Third-party imports ---
from collections import Counter 
import pandas as pd

# --- Local imports ---
from core.utils.logger import get_logger

log = get_logger(__name__)

# Common file extensions
DEFAULT_FILE_EXTS: Set[str] = {
    ".csv", ".tsv", ".txt",
    ".xlsx", ".xls",
    ".parquet",
    ".json", ".yaml", ".yml",
    ".joblib", ".pkl", ".pickle" 
}

def ensure_directories(
    dirs: Union[str, Path, Iterable[Union[str, Path]]],
    *,
    known_file_exts: Optional[Set[str]] = None,
    strict: bool = False,
) -> None:
    """
    Check and create one or more directories safely.

    If a path appears to be a file (based on its extension or existing file),
    only the parent directory will be created. If `strict=True`, an error is raised
    when a path is provided with extension.

    Args:
        dirs (Union[str, Path, Iterable[Union[str, Path]]]): 
            The directory or directories to create. Can be a string, Path, or an iterable of paths.
        
        known_file_exts (Optional[Set[str]], optional): 
            A set of known file extensions. Used to identify file-like paths (default None).

        strict (bool): 
            If True, raises an error when a file-like path is passed. (default False).
            
    Returns:
        None
    """
    if known_file_exts is None:
        known_file_exts = DEFAULT_FILE_EXTS

    # If string or Path add into a list
    if isinstance(dirs, (str, Path)):
        dirs = [dirs]

    for d in dirs:
        p = Path(d).expanduser()

        # If it already exists, ensure the correct parent and continue
        if p.exists():
            if p.is_dir():
                continue
            if p.is_file():
                p.parent.mkdir(parents=True, exist_ok=True)
                continue
            # symlink/special: safest to ensure parent
            p.parent.mkdir(parents=True, exist_ok=True)
            continue

        # Heuristic: treat as file if it has a known file extension
        file_like_path = any(s.lower() in known_file_exts for s in p.suffixes)

        if file_like_path:
            if strict:
                raise IsADirectoryError(f"Expected a directory path, got a file-like path: {p}")
            p.parent.mkdir(parents=True, exist_ok=True)
        else:
            p.mkdir(parents=True, exist_ok=True)

def find_path(filename: str, directory: str) -> str:
    """
    Find the full path of a file by name inside a given directory.

    Args:
        filename (str): The name of the file.
        directory (str): Directory where the file stored.
    
    Returns:
        str: Full path of the file.
    """
    directory = Path(directory)
    # Ensure the directory exits or not
    if not os.path.isdir(directory):
        log.error("Directory does not exist: %s", directory)
        raise NotADirectoryError(f"Directory does not exist: {directory}")

    for fname in os.listdir(directory):
        if os.path.splitext(fname)[0] == filename: 
            full_path = Path(os.path.join(directory, fname))
            log.info("Found file: %s", full_path)
            return full_path

    log.error("No file named '%s' found in %s", filename, directory)
    raise FileNotFoundError(f"No file named '{filename}' found in {directory}")

def highlight_common(val: Any, col_counter: Dict[str, int], color: str = 'darkgreen') -> str:
        """
        Returns a CSS style string to highlight common values based on col_counter.

        Args:
            val (Union[str, float]): The value to be checked for highlighting.
            col_counter (Dict[str, int]): Columns with common counting.

        Returns:
            str: CSS style string for DataFrame styling.
        """
        if pd.isnull(val):
            return ""
        return f'background-color: {color}' if col_counter[val] > 1 else ""

# Helper function for comparing DataFrames
def compare_dataframes(dir_path: str) -> Tuple[pd.DataFrame, Counter]:
    """
    Analyze files or data in the given directory and return a comparison DataFrame and 
    a Counter of column occurrences.

    Args:
        dir_path (str): Path to the directory to analyze.

    Returns:
        Tuple[pd.DataFrame, Counter]: A DataFrame comparing the relevant data extracted from files, and A Counter object containing counts of each column or key encountered.
    """
    dir_path = Path(dir_path)
    
    if not os.path.isdir(dir_path):
        log.error("Directory does not exist: %s", dir_path)
        raise NotADirectoryError(f"Directory does not exist: {dir_path}")
    
    # Collect all columns
    column_lists = {}
    for file in os.listdir(dir_path):
        if file.endswith('.csv'):         # Only CSV files
            df_name = file.split('.')[0]
            data = pd.read_csv(os.path.join(dir_path, file))
            column_lists[df_name] = list(data.columns)

    # Counts 'unique column names'
    all_columns_flat = [col for cols in column_lists.values() for col in cols]
    col_counter = Counter(all_columns_flat)

    # Count 'shared columns' each dataset
    dataset_common_counts = {
        name: sum(1 for col in cols if col_counter[col] > 1)
        for name, cols in column_lists.items()
    }

    # Sort datasets by 'common column' count (descending)
    sorted_datasets = sorted(column_lists.keys(), key= lambda name: dataset_common_counts[name], reverse= True)

    # Column lists into equal length by adding None
    max_len = max(len(column_lists[name]) for name in sorted_datasets)
    for name in column_lists:
        column_lists[name] += [None] * (max_len - len(column_lists[name]))

    # Create the final dataset
    compare_df = pd.DataFrame({name: column_lists[name] for name in sorted_datasets})

    return compare_df, col_counter

