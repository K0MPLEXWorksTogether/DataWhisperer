"""
Utility functions for DataWhisperer.

This module contains helper functions for problem type detection
and other common operations.
"""

import dask.dataframe as dd
import numpy as np
from pandas.api.types import is_numeric_dtype


def detect_problem_type(dataframe: dd.DataFrame, target_index: int) -> str:
    """
    Auto-detect whether the problem is regression or classification
    based on target characteristics.
    
    Logic:
    - If target is not numeric -> classification
    - If target is numeric:
        - If unique_count < 20 and unique_ratio < 0.05 -> classification
        - Otherwise -> regression
    
    :param dataframe: The input Dask DataFrame.
    :param target_index: The index of the target column.
    :return: 'regression', 'binary', or 'multiclass'
    """
    try:
        # Get target column
        target_column = dataframe.columns[target_index]
        target_series = dataframe[target_column].compute()
        
        # Check if numeric
        if not is_numeric_dtype(target_series):
            # Categorical target -> classification
            unique_count = target_series.nunique()
            if unique_count == 2:
                return "binary"
            else:
                return "multiclass"
        
        # Numeric target - need to determine if discrete classes or continuous
        unique_count = target_series.nunique()
        total_count = len(target_series)
        unique_ratio = unique_count / total_count if total_count > 0 else 0
        
        # If very few unique values and low ratio -> likely classification
        if unique_count < 20 and unique_ratio < 0.05:
            if unique_count == 2:
                return "binary"
            else:
                return "multiclass"
        
        # Check if values are all integers (another hint for classification)
        if unique_ratio < 0.1 and np.all(target_series == target_series.astype(int)):
            if unique_count == 2:
                return "binary"
            elif unique_count < 20:
                return "multiclass"
        
        # Otherwise, continuous target -> regression
        return "regression"
        
    except Exception as e:
        # Default to classification if uncertain
        print(f"Error detecting problem type: {e}. Defaulting to multiclass.")
        return "multiclass"


def is_regression(problem_type: str) -> bool:
    """
    Check if the problem type is regression.
    
    :param problem_type: The problem type string.
    :return: True if regression, False otherwise.
    """
    return problem_type == "regression"


def is_classification(problem_type: str) -> bool:
    """
    Check if the problem type is classification (binary or multiclass).
    
    :param problem_type: The problem type string.
    :return: True if classification, False otherwise.
    """
    return problem_type in ["binary", "multiclass"]
