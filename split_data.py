import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def split_data(file):
    # Read CSV
    data = pd.read_csv(file)

    # Remove irrelevant columns
    cols_to_remove = [
        'farm_id', 'sensor_id', 'timestamp',
        'sowing_date', 'harvest_date'
    ]
    data = data.drop(columns=[col for col in cols_to_remove if col in data.columns])

    # Encode categorical variables
    categorical_vars = [
        'region', 'crop_type', 'irrigation_type',
        'fertilizer_type', 'crop_disease_status'
    ]
    for col in categorical_vars:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # Drop missing rows
    data = data.dropna()

    # Separate target from features
    target = data['yield_kg_per_hectare'].values
    features = data.drop(columns=['yield_kg_per_hectare']).values

    # Shuffle the data
    np.random.seed(42)
    indices = np.random.permutation(len(features))
    split_point = int(0.8 * len(features))
    train_idx, test_idx = indices[:split_point], indices[split_point:]

    # Train/test split
    train_features, test_features = features[train_idx], features[test_idx]
    train_target, test_target = target[train_idx], target[test_idx]

    # Z-score normalization
    scaler_features = StandardScaler()
    train_features = scaler_features.fit_transform(train_features)
    test_features = scaler_features.transform(test_features)

    scaler_target = StandardScaler()
    train_target = scaler_target.fit_transform(train_target.reshape(-1, 1)).flatten()
    test_target = scaler_target.transform(test_target.reshape(-1, 1)).flatten()

    # Concatenate features and target
    train_data = np.hstack((train_features, train_target.reshape(-1, 1)))
    test_data = np.hstack((test_features, test_target.reshape(-1, 1)))

    return train_data, test_data
