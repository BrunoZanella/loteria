import pandas as pd
import numpy as np
import random
from .feature_engineering import create_features
from .model_training import train_prediction_model
from .pattern_analysis import analyze_patterns, evaluate_combination
from .frequency_analysis import analyze_frequency, get_hot_numbers
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import random
from typing import List, Tuple, Dict
import logging


def prepare_lottery_data(df):
    """Prepare lottery data for analysis"""
    ball_columns = [col for col in df.columns if col.startswith('Bola')]
    if not ball_columns:
        raise ValueError("No valid number columns found")
    
    numbers_df = df[ball_columns].apply(pd.to_numeric, errors='coerce')
    return numbers_df

def generate_candidate_numbers(model, features, game, hot_numbers, patterns):
    """Generate candidate numbers based on model prediction"""
    candidates = []
    prediction = model.predict(features)[0]  # Get base probability
    
    # Create probability distribution
    probabilities = np.ones(game.total_numbers)
    
    # Convert prediction to a scaling factor if it's not an integer
    scaling_factor = max(0.1, min(2.0, prediction / probabilities.mean()))
    probabilities = probabilities * scaling_factor
    
    # Adjust probabilities based on hot numbers
    for num, freq in hot_numbers:
        if num <= game.total_numbers:
            idx = int(num - 1)  # Ensure integer index
            probabilities[idx] *= (1 + freq/100)
    
    # Normalize probabilities
    probabilities = probabilities / probabilities.sum()
    
    # Generate numbers
    numbers = []
    while len(numbers) < game.numbers_to_choose:
        temp_probs = probabilities.copy()
        
        # Zero out already selected numbers using integer indices
        if numbers:
            selected_indices = [int(n - 1) for n in numbers]
            temp_probs[selected_indices] = 0
        
        # Renormalize if not all zeros
        if temp_probs.sum() > 0:
            temp_probs = temp_probs / temp_probs.sum()
        else:
            # If all probabilities are zero, reset with uniform distribution
            remaining_numbers = set(range(1, game.total_numbers + 1)) - set(numbers)
            if remaining_numbers:
                next_number = random.choice(list(remaining_numbers))
                numbers.append(next_number)
                continue
        
        # Generate next number
        number = np.random.choice(range(1, game.total_numbers + 1), p=temp_probs)
        numbers.append(int(number))  # Ensure integer
    
    numbers.sort()
    score = evaluate_combination(numbers, patterns)
    return numbers, score




'''
def generate_ai_numbers(game, num_tickets=1):
    """
    Generate AI-powered lottery number predictions with diversification
    Returns predictions in the same format as the original function
    """
    try:
        # Configuração de logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        
        # Leitura e preparação dos dados
        df = pd.read_csv(game.historical_data.path)
        numbers_df = prepare_enhanced_lottery_data(df)
        
        # Criar features avançadas
        X, y = create_advanced_features(numbers_df)
        
        # Treinar ensemble de modelos
        models = train_ensemble_models(X, y)
        
        # Analisar padrões e frequências com técnicas avançadas
        patterns = analyze_enhanced_patterns(numbers_df)
        frequency_dict = analyze_advanced_frequency(numbers_df.values, game.total_numbers)
        hot_numbers = get_smart_hot_numbers(frequency_dict, limit=20)
        
        # Gerar previsões usando ensemble
        predictions = []
        attempts = 0
        max_attempts = num_tickets * 10

        while len(predictions) < num_tickets and attempts < max_attempts:
            # Gerar previsão usando ensemble
            X_last = X.iloc[-1:] if len(X) > 0 else X.iloc[:1]
            X_last_scaled = models['scaler'].transform(X_last)
            
            rf_pred = models['rf'].predict(X_last_scaled)
            gb_pred = models['gb'].predict(X_last_scaled)
            xgb_pred = models['xgb'].predict(X_last_scaled)
            
            # Combinar previsões
            weights = np.array([0.4, 0.3, 0.3])
            combined_pred = np.average([rf_pred, gb_pred, xgb_pred], weights=weights, axis=0)
            
            # Gerar números candidatos
            numbers = generate_smart_numbers(
                combined_pred[0], 
                game.total_numbers, 
                game.numbers_to_choose,
                hot_numbers
            )
            
            # Verificar diversidade
            if is_diverse_prediction(numbers, predictions, game.numbers_to_choose):
                predictions.append(sorted(numbers))
            
            attempts += 1
        
        # Preencher slots restantes se necessário
        while len(predictions) < num_tickets:
            numbers = sorted(random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose))
            if is_diverse_prediction(numbers, predictions, game.numbers_to_choose):
                predictions.append(numbers)
        
        return predictions[0] if num_tickets == 1 else predictions

    except Exception as e:
        logger.error(f"Error in AI generation: {str(e)}")
        return fallback_generation(game, num_tickets)

def prepare_enhanced_lottery_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preparação aprimorada dos dados históricos"""
    # Assume que o DataFrame contém as colunas com os números sorteados
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return df[numeric_columns]

def create_advanced_features(numbers_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Criação de features mais sofisticadas"""
    features = pd.DataFrame()
    
    # Features estatísticas
    for window in [3, 5, 10]:
        features[f'mean_{window}'] = numbers_df.rolling(window=window).mean().mean(axis=1)
        features[f'std_{window}'] = numbers_df.rolling(window=window).std().mean(axis=1)
        
    # Lag features
    for lag in [1, 2, 3]:
        lagged = numbers_df.shift(lag)
        features[f'lag_{lag}'] = lagged.mean(axis=1)
    
    features = features.fillna(0)
    
    return features.iloc[:-1], numbers_df.iloc[1:].values

def train_ensemble_models(X: pd.DataFrame, y: np.ndarray) -> Dict:
    """Treina um ensemble de modelos de ML"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    # Treinar modelos
    rf_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)
    xgb_model.fit(X_scaled, y)
    
    return {
        'rf': rf_model,
        'gb': gb_model,
        'xgb': xgb_model,
        'scaler': scaler
    }

def analyze_enhanced_patterns(numbers_df: pd.DataFrame) -> Dict:
    """Análise de padrões nos números"""
    patterns = {
        'mean': numbers_df.mean().mean(),
        'std': numbers_df.std().mean(),
        'min': numbers_df.min().min(),
        'max': numbers_df.max().max()
    }
    return patterns

def analyze_advanced_frequency(numbers: np.ndarray, total_numbers: int) -> Dict:
    """Análise de frequência com pesos temporais"""
    frequency_dict = {i: 0 for i in range(1, total_numbers + 1)}
    weights = np.linspace(0.5, 1.0, len(numbers))
    
    for idx, row in enumerate(numbers):
        for number in row.flatten():
            if number > 0:  # Ignora zeros
                frequency_dict[int(number)] += weights[idx]
    
    return frequency_dict

def get_smart_hot_numbers(frequency_dict: Dict, limit: int = 20) -> List[int]:
    """Seleção de números quentes baseada na frequência"""
    return [num for num, _ in sorted(
        frequency_dict.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:limit]]

def generate_smart_numbers(
    pred: np.ndarray,
    total_numbers: int,
    numbers_to_choose: int,
    hot_numbers: List[int]
) -> List[int]:
    """Geração inteligente de números"""
    # Combinar previsão do modelo com números quentes
    numbers_pool = set(range(1, total_numbers + 1))
    hot_set = set(hot_numbers[:numbers_to_choose])
    
    # Escolher alguns números dos hot numbers
    hot_count = random.randint(numbers_to_choose // 3, numbers_to_choose // 2)
    selected = set(random.sample(hot_set, hot_count))
    
    # Completar com números aleatórios
    remaining = numbers_to_choose - len(selected)
    available = list(numbers_pool - selected)
    selected.update(random.sample(available, remaining))
    
    return sorted(list(selected))

def is_diverse_prediction(
    numbers: List[int],
    existing_predictions: List[List[int]],
    numbers_to_choose: int
) -> bool:
    """Verifica se a previsão é suficientemente diversa"""
    if not existing_predictions:
        return True
    
    numbers_set = set(numbers)
    return all(
        len(numbers_set.intersection(set(existing))) < numbers_to_choose // 2
        for existing in existing_predictions
    )

def fallback_generation(game, num_tickets: int) -> List[List[int]]:
    """Geração aleatória como fallback"""
    predictions = []
    while len(predictions) < num_tickets:
        numbers = sorted(random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose))
        if numbers not in predictions:
            predictions.append(numbers)
    return predictions[0] if num_tickets == 1 else predictions


'''





##################################################################
####       versao 
####         2
###################################################################








import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import joblib

def generate_ai_numbers(game, num_tickets=1):
    """
    Gera números de loteria previstos usando um modelo treinado previamente.
    """
    try:
        # Caminho do modelo treinado
        model_path = f"trained_models/{game.name.lower()}_rf_model.pkl"
        scaler_path = f"trained_models/{game.name.lower()}_scaler.pkl"

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Modelo ou escalador não encontrados para {game.name}.")

        # Carregar dados históricos
        df = pd.read_csv(game.historical_data.path)
        numbers_df = prepare_enhanced_lottery_data(df)

        # Criar features avançadas
        X, _ = create_advanced_features(numbers_df)

        # Escalar os dados
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)

        # Carregar modelo e fazer previsões
        model = joblib.load(model_path)
        predictions = model.predict(X_scaled[-1:])

        # Gerar números baseados nas previsões
        numbers = generate_smart_numbers(
            predictions,
            game.total_numbers,
            game.numbers_to_choose,
            []
        )

        return sorted(numbers)

    except Exception as e:
        print(f"Erro na geração de números: {str(e)}")
        return fallback_generation(game, num_tickets)




def prepare_enhanced_lottery_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preparação aprimorada dos dados históricos"""
    # Assume que o DataFrame contém as colunas com os números sorteados
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return df[numeric_columns]

def create_advanced_features(numbers_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Criação de features mais sofisticadas"""
    features = pd.DataFrame()

    # Features estatísticas
    for window in [3, 5, 10]:
        features[f'mean_{window}'] = numbers_df.rolling(window=window).mean().mean(axis=1)
        features[f'std_{window}'] = numbers_df.rolling(window=window).std().mean(axis=1)

    # Lag features
    for lag in [1, 2, 3]:
        lagged = numbers_df.shift(lag)
        features[f'lag_{lag}'] = lagged.mean(axis=1)

    # Preencher valores ausentes
    features = features.fillna(0)

    # Definir y como a soma das colunas na próxima linha
    y = numbers_df.iloc[1:].sum(axis=1).values  # Aqui y é a soma das colunas da próxima linha

    return features.iloc[:-1], y


def train_ensemble_models(X: pd.DataFrame, y: np.ndarray) -> Dict:
    """Treina um ensemble de modelos de ML"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    # Treinar modelos
    rf_model.fit(X_scaled, y)
    gb_model.fit(X_scaled, y)
    xgb_model.fit(X_scaled, y)
    
    return {
        'rf': rf_model,
        'gb': gb_model,
        'xgb': xgb_model,
        'scaler': scaler
    }

def analyze_enhanced_patterns(numbers_df: pd.DataFrame) -> Dict:
    """Análise de padrões nos números"""
    patterns = {
        'mean': numbers_df.mean().mean(),
        'std': numbers_df.std().mean(),
        'min': numbers_df.min().min(),
        'max': numbers_df.max().max()
    }
    return patterns

def analyze_advanced_frequency(numbers: np.ndarray, total_numbers: int) -> Dict:
    """Análise de frequência com pesos temporais"""
    frequency_dict = {i: 0 for i in range(1, total_numbers + 1)}
    weights = np.linspace(0.5, 1.0, len(numbers))
    
    for idx, row in enumerate(numbers):
        for number in row.flatten():
            if number > 0:  # Ignora zeros
                frequency_dict[int(number)] += weights[idx]
    
    return frequency_dict

def get_smart_hot_numbers(frequency_dict: Dict, limit: int = 20) -> List[int]:
    """Seleção de números quentes baseada na frequência"""
    return [num for num, _ in sorted(
        frequency_dict.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:limit]]
    
    
def generate_smart_numbers(
    pred: np.ndarray,
    total_numbers: int,
    numbers_to_choose: int,
    hot_numbers: List[int]
) -> List[int]:
    """Geração inteligente de números"""
    # Criar o conjunto completo de números
    numbers_pool = set(range(1, total_numbers + 1))
    hot_set = set(hot_numbers[:numbers_to_choose])
    
    # Escolher alguns números dos hot numbers
    hot_count = min(len(hot_set), random.randint(numbers_to_choose // 3, numbers_to_choose // 2))
    selected = set(random.sample(list(hot_set), hot_count))
    
    # Completar com números aleatórios
    remaining = numbers_to_choose - len(selected)
    available = list(numbers_pool - selected)  # Converter para lista
    if remaining > 0:
        selected.update(random.sample(available, min(remaining, len(available))))
    
    return sorted(list(selected))


'''
def generate_smart_numbers(
    pred: np.ndarray,
    total_numbers: int,
    numbers_to_choose: int,
    hot_numbers: List[int]
) -> List[int]:
    """Geração inteligente de números"""
    # Combinar previsão do modelo com números quentes
    numbers_pool = set(range(1, total_numbers + 1))
    hot_set = set(hot_numbers[:numbers_to_choose])
    
    # Escolher alguns números dos hot numbers
    hot_count = random.randint(numbers_to_choose // 3, numbers_to_choose // 2)
    selected = set(random.sample(hot_set, hot_count))
    
    # Completar com números aleatórios
    remaining = numbers_to_choose - len(selected)
    available = list(numbers_pool - selected)
    selected.update(random.sample(available, remaining))
    
    return sorted(list(selected))
'''
def is_diverse_prediction(
    numbers: List[int],
    existing_predictions: List[List[int]],
    numbers_to_choose: int
) -> bool:
    """Verifica se a previsão é suficientemente diversa"""
    if not existing_predictions:
        return True
    
    numbers_set = set(numbers)
    return all(
        len(numbers_set.intersection(set(existing))) < numbers_to_choose // 2
        for existing in existing_predictions
    )

def fallback_generation(game, num_tickets: int) -> List[List[int]]:
    """Geração aleatória como fallback"""
    predictions = []
    while len(predictions) < num_tickets:
        numbers = sorted(random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose))
        if numbers not in predictions:
            predictions.append(numbers)
    return predictions[0] if num_tickets == 1 else predictions


















# def generate_ai_numbers(game, num_tickets=1):
#     """Generate AI-powered lottery number predictions with diversification"""
#     try:
#         # Read and prepare data
#         df = pd.read_csv(game.historical_data.path)
#         numbers_df = prepare_lottery_data(df)

#         # Create features and prepare training data
#         X = create_features(numbers_df.iloc[:-1])
#         y = numbers_df.iloc[1:].values

#         # Apply recency weights
#         sample_weights = np.linspace(0.1, 2.0, len(X))

#         # Train model
#         model = train_prediction_model(X, y, sample_weights)

#         # Analyze patterns and frequencies
#         patterns = analyze_patterns(numbers_df)
#         frequency_dict = analyze_frequency(numbers_df.values.tolist(), game.total_numbers)
#         hot_numbers = get_hot_numbers(frequency_dict, limit=15)

#         # Generate predictions
#         last_features = create_features(numbers_df.iloc[-1:])
#         predictions = []
#         attempts = 0
#         max_attempts = num_tickets * 5  # Increased attempts to ensure diversity

#         while len(predictions) < num_tickets and attempts < max_attempts:
#             numbers, score = generate_candidate_numbers(
#                 model, last_features, game, hot_numbers, patterns
#             )
#             numbers_set = set(numbers)

#             # Ensure prediction is unique and diverse
#             is_diverse = all(
#                 len(numbers_set.intersection(set(existing))) < game.numbers_to_choose // 2
#                 for existing in predictions
#             )
#             if is_diverse and numbers not in predictions:
#                 predictions.append(numbers)

#             attempts += 1

#         # Fill remaining slots with random numbers if needed
#         while len(predictions) < num_tickets:
#             numbers = sorted(random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose))
#             numbers_set = set(numbers)
#             is_diverse = all(
#                 len(numbers_set.intersection(set(existing))) < game.numbers_to_choose // 2
#                 for existing in predictions
#             )
#             if is_diverse and numbers not in predictions:
#                 predictions.append(numbers)

#         return predictions[0] if num_tickets == 1 else predictions

#     except Exception as e:
#         print(f"Error in AI generation: {str(e)}")
#         # Fallback to random generation
#         predictions = []
#         while len(predictions) < num_tickets:
#             numbers = sorted(random.sample(range(1, game.total_numbers + 1), game.numbers_to_choose))
#             if numbers not in predictions:
#                 predictions.append(numbers)
#         return predictions[0] if num_tickets == 1 else predictions











'''

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

'''