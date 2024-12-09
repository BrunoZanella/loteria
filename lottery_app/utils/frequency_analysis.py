from collections import Counter
import numpy as np
import pandas as pd

def analyze_frequency(numbers_list, total_numbers):
    """Analyze frequency of numbers in the dataset"""
    all_numbers = []
    for numbers in numbers_list:
        if isinstance(numbers, (list, np.ndarray)):
            all_numbers.extend([n for n in numbers if not pd.isna(n)])
    
    frequency = Counter(all_numbers)
    return {i: frequency.get(i, 0) for i in range(1, total_numbers + 1)}

def get_hot_numbers(frequency_dict, limit=10):
    """Get most frequent numbers"""
    return sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)[:limit]

def calculate_similarity(numbers1, numbers2):
    """Calculate Jaccard similarity between two sets of numbers"""
    set1 = set(numbers1)
    set2 = set(numbers2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0