import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import random
from .data_preparation import prepare_historical_data, get_last_year_data
from .frequency_analysis import analyze_frequency, get_hot_numbers, calculate_similarity
from .pattern_analysis import analyze_patterns, get_common_patterns, evaluate_combination

def create_features(numbers_df):
    """Create additional features for model training"""
    features = []
    for row in numbers_df.values:
        sorted_nums = sorted(row)
        feature = [
            np.mean(sorted_nums),
            np.std(sorted_nums),
            max(sorted_nums) - min(sorted_nums),
            sum(sorted_nums),
            *sorted_nums
        ]
        features.append(feature)
    return np.array(features)

def train_models(X, y, sample_weights):
    """Train multiple models for ensemble prediction"""
    models = []
    
    # Random Forest model
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=None
    )
    rf_model.fit(X, y, sample_weight=sample_weights)
    models.append(rf_model)
    
    # Gradient Boosting model
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=8,
        random_state=None
    )
    gb_model.fit(X, y, sample_weight=sample_weights)
    models.append(gb_model)
    
    return models

def generate_candidate_numbers(models, last_numbers, game, hot_numbers, patterns):
    """Generate candidate numbers using ensemble prediction"""
    candidates = []
    
    for model in models:
        for _ in range(3):  # Generate multiple predictions per model
            prediction = model.predict(last_numbers)[0]
            numbers = []
            
            # Use predicted values as probabilities
            probabilities = np.zeros(game.total_numbers)
            for i, prob in enumerate(prediction[:game.total_numbers]):
                probabilities[i] = max(0, min(1, prob / 100))
            
            # Adjust probabilities based on hot numbers
            for num, freq in hot_numbers:
                if num <= len(probabilities):
                    probabilities[num-1] *= (1 + freq/100)
            
            # Generate numbers based on adjusted probabilities
            while len(numbers) < game.numbers_to_choose:
                probs = probabilities.copy()
                probs[np.array(numbers) - 1] = 0  # Zero out already selected numbers
                probs = probs / probs.sum()  # Normalize
                
                num = np.random.choice(range(1, game.total_numbers + 1), p=probs)
                numbers.append(num)
            
            numbers = sorted(numbers)
            score = evaluate_combination(numbers, patterns)
            candidates.append((numbers, score))
    
    return candidates

def select_best_combinations(candidates, num_tickets, similarity_threshold=0.6):
    """Select best combinations based on pattern scores and uniqueness"""
    selected = []
    candidates.sort(key=lambda x: x[1], reverse=True)  # Sort by score
    
    for numbers, score in candidates:
        if len(selected) >= num_tickets:
            break
            
        is_unique = True
        for existing in selected:
            if calculate_similarity(numbers, existing) > similarity_threshold:
                is_unique = False
                break
                
        if is_unique:
            selected.append(numbers)
    
    return selected

def generate_ai_numbers(game, num_tickets=1):
    """Generate multiple unique AI predictions for lottery tickets"""
    try:
        # Read and prepare data
        df = pd.read_csv(game.historical_data.path)
        numbers_df, _ = prepare_historical_data(df)
        last_year_data = get_last_year_data(df, numbers_df)
        
        # Analyze patterns and frequencies
        patterns = analyze_patterns(last_year_data)
        frequency_dict = analyze_frequency(last_year_data.values.tolist(), game.total_numbers)
        hot_numbers = get_hot_numbers(frequency_dict, limit=15)
        
        # Create enhanced features
        X = create_features(numbers_df.iloc[:-1])
        y = numbers_df.iloc[1:].values
        
        # Apply recency weights
        sample_weights = np.linspace(0.1, 2.0, len(X))
        
        # Train ensemble models
        models = train_models(X, y, sample_weights)
        
        # Generate candidates using ensemble prediction
        last_features = create_features(numbers_df.iloc[-1:])
        candidates = generate_candidate_numbers(
            models, last_features, game, hot_numbers, patterns
        )
        
        # Select best combinations
        selected_combinations = select_best_combinations(
            candidates, num_tickets, similarity_threshold=0.6
        )
        
        if len(selected_combinations) < num_tickets:
            remaining = num_tickets - len(selected_combinations)
            while len(selected_combinations) < num_tickets:
                numbers = random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose)
                if all(calculate_similarity(numbers, existing) <= 0.6 
                      for existing in selected_combinations):
                    selected_combinations.append(sorted(numbers))
        
        return selected_combinations[0] if num_tickets == 1 else selected_combinations
        
    except Exception as e:
        print(f"Error in AI generation: {str(e)}")
        # Generate unique random combinations if AI fails
        predictions = []
        while len(predictions) < num_tickets:
            numbers = random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose)
            if numbers not in predictions:
                predictions.append(sorted(numbers))
        return predictions[0] if num_tickets == 1 else predictions