import numpy as np
from itertools import combinations
from collections import defaultdict

def analyze_patterns(numbers_df):
    """Analyze patterns in winning combinations"""
    patterns = defaultdict(int)
    
    # Analyze number gaps
    for row in numbers_df.values:
        sorted_nums = sorted(row)
        gaps = [sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1)]
        patterns[f'gaps_{tuple(gaps)}'] += 1
    
    # Analyze number ranges distribution
    for row in numbers_df.values:
        ranges = [0, 0, 0]  # low (1-20), mid (21-40), high (41-60)
        for num in row:
            if num <= 20:
                ranges[0] += 1
            elif num <= 40:
                ranges[1] += 1
            else:
                ranges[2] += 1
        patterns[f'ranges_{tuple(ranges)}'] += 1
    
    return patterns

def get_common_patterns(patterns, top_n=5):
    """Get most common patterns"""
    return sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]

def evaluate_combination(numbers, patterns):
    """Evaluate how well a combination matches common patterns"""
    score = 0
    sorted_nums = sorted(numbers)
    
    # Check gaps pattern
    gaps = tuple(sorted_nums[i+1] - sorted_nums[i] for i in range(len(sorted_nums)-1))
    score += patterns.get(f'gaps_{gaps}', 0)
    
    # Check ranges pattern
    ranges = [0, 0, 0]
    for num in numbers:
        if num <= 20:
            ranges[0] += 1
        elif num <= 40:
            ranges[1] += 1
        else:
            ranges[2] += 1
    score += patterns.get(f'ranges_{tuple(ranges)}', 0)
    
    return score