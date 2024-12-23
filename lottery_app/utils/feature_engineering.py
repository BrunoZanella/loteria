import numpy as np
import pandas as pd

def create_features(numbers_df):
    """Create features for each draw"""
    features = []
    for row in numbers_df.values:
        sorted_nums = np.sort(row)
        features.append([
            np.mean(sorted_nums),           # Mean of numbers
            np.std(sorted_nums),            # Standard deviation
            np.max(sorted_nums) - np.min(sorted_nums),  # Range
            np.sum(sorted_nums),            # Sum
            len(np.unique(sorted_nums)),    # Unique numbers
            np.median(sorted_nums)          # Median
        ])
    return np.array(features)