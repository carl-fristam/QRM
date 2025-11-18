import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split



def preprocess_features(df, target_col='Class'):
    """
    Preprocess credit card fraud dataset.
    - Scale Amount and Time using RobustScaler (robust to outliers)
    - V1-V28 are already PCA-transformed and scaled
    """
    df = df.copy()
    
    # Check if Amount and Time columns exist
    if 'Amount' in df.columns:
        scaler = RobustScaler()
        df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
        df = df.drop('Amount', axis=1)
        print("✓ Amount scaled and replaced with Amount_scaled")
    
    if 'Time' in df.columns:
        scaler = RobustScaler()
        df['Time_scaled'] = scaler.fit_transform(df[['Time']])
        df = df.drop('Time', axis=1)
        print("✓ Time scaled and replaced with Time_scaled")
    
    return df

def prepare_data(df, target_col='Class', test_size=0.3, random_state=42, 
                 preprocess=True):
    """
    Split data into train/test sets with stratification.
    Keep test set untouched for final evaluation.
    
    Parameters:
    -----------
    df : DataFrame
        Your fraud dataset
    target_col : str
        Name of the target column
    test_size : float
        Proportion of data for testing
    random_state : int
        Random seed
    preprocess : bool
        Whether to scale Amount and Time (default: True)
    """
    # Preprocess if needed
    if preprocess:
        print("\n" + "="*60)
        print("PREPROCESSING FEATURES")
        print("="*60)
        df = preprocess_features(df, target_col)
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Fraud cases: {y.sum()} ({y.mean()*100:.2f}%)")
    print(f"Non-fraud cases: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
    
    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud in train: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Fraud in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    
    return X_train, X_test, y_train, y_test