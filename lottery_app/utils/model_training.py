from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_prediction_model(X, y, sample_weights):
    """Train a single prediction model"""
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Reshape y to be 1-dimensional if needed
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # Use mean as target
        
    model.fit(X, y, sample_weight=sample_weights)
    return model