import pandas as pd
from datetime import datetime, timedelta

def prepare_historical_data(df):
    """Prepare and clean historical data for Brazilian lottery formats"""
    ball_columns = [col for col in df.columns if col.startswith('Bola')]
    
    if not ball_columns:
        raise ValueError("No valid number columns (Bola1, Bola2, etc.) found in the dataset")
    
    ball_columns.sort()
    
    if 'Data Sorteio' in df.columns:
        df['Data Sorteio'] = pd.to_datetime(df['Data Sorteio'], format='%d/%m/%Y', errors='coerce')
    elif 'Data do Sorteio' in df.columns:
        df['Data do Sorteio'] = pd.to_datetime(df['Data do Sorteio'], format='%d/%m/%Y', errors='coerce')
    
    numbers_df = df[ball_columns].apply(pd.to_numeric, errors='coerce')
    
    return numbers_df, ball_columns

def get_last_year_data(df, numbers_df):
    """Extract the last year of lottery data"""
    if 'Data Sorteio' in df.columns:
        last_date = pd.to_datetime(df['Data Sorteio'].iloc[-1], format='%d/%m/%Y')
        year_ago = last_date - timedelta(days=365)
        last_year_mask = pd.to_datetime(df['Data Sorteio'], format='%d/%m/%Y') >= year_ago
        return numbers_df[last_year_mask]
    elif 'Data do Sorteio' in df.columns:
        last_date = pd.to_datetime(df['Data do Sorteio'].iloc[-1], format='%d/%m/%Y')
        year_ago = last_date - timedelta(days=365)
        last_year_mask = pd.to_datetime(df['Data do Sorteio'], format='%d/%m/%Y') >= year_ago
        return numbers_df[last_year_mask]
    else:
        return numbers_df.tail(52)  # Assuming weekly drawings