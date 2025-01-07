import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Function to load and process demand data
def load_demand_data(file_path):
    # Read all lines first to find where the actual data starts
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Find the actual header row (should contain Date and Hour)
    header_idx = None
    for i, line in enumerate(lines):
        if all(col in line for col in ['Date', 'Hour']):
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError(f"Could not find header row in {file_path}")
    
    # Read CSV starting from the header row
    df = pd.read_csv(file_path, skiprows=header_idx)
    
    # Clean column names (remove any whitespace or special characters)
    df.columns = df.columns.str.strip()
    
    # Convert date and hour columns
    date_col = df.columns[0]  # First column should be Date
    hour_col = df.columns[1]  # Second column should be Hour
    demand_col = 'Ontario Demand'  # Use Ontario Demand column
    
    # Convert columns with enhanced features
    df['date'] = pd.to_datetime(df[date_col])
    df['hour'] = df[hour_col].astype(int)
    df['demand'] = df[demand_col].astype(float)
    
    # Time-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_winter'] = df['month'].isin([12, 1, 2])
    df['is_summer'] = df['month'].isin([6, 7, 8])
    
    # Calculate peak hours (typically business hours)
    df['is_peak_hour'] = df['hour'].between(9, 17)
    
    # Group by date to get daily features
    daily_stats = df.groupby(df['date'].dt.date).agg({
        'demand': ['mean', 'max', 'min', 'std']
    }).reset_index()
    daily_stats.columns = ['date', 'daily_mean_demand', 'daily_max_demand', 
                          'daily_min_demand', 'daily_demand_std']
    
    # Merge daily stats back
    df = df.merge(daily_stats, left_on=df['date'].dt.date, right_on='date', 
                 suffixes=('', '_daily'))
    
    return df

# Function to load and process weather data
def load_weather_data(file_paths):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path, dtype={
            'Date/Time': str,
            'Max Temp (°C)': str,
            'Min Temp (°C)': str
        })
        dfs.append(df)
    
    weather_df = pd.concat(dfs)
    
    # Clean and convert columns
    weather_df['Date/Time'] = pd.to_datetime(weather_df['Date/Time'].str.strip('"'))
    
    # Convert temperature columns, replacing 'M' with NaN
    weather_df['Max Temp (°C)'] = pd.to_numeric(
        weather_df['Max Temp (°C)'].str.strip('"').replace('M', np.nan),
        errors='coerce'
    )
    weather_df['Min Temp (°C)'] = pd.to_numeric(
        weather_df['Min Temp (°C)'].str.strip('"').replace('M', np.nan),
        errors='coerce'
    )
    
    weather_df['day_of_year'] = weather_df['Date/Time'].dt.dayofyear
    weather_df['day_of_week'] = weather_df['Date/Time'].dt.dayofweek
    return weather_df

try:
    # Load 2024 weather data
    print("Loading weather data...")
    weather_files = [f'2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_{str(i).zfill(2)}.csv' for i in range(1, 9)]
    weather_2024 = load_weather_data(weather_files)

    # Load historical demand data
    print("Loading 2021 demand data...")
    demand_2021 = load_demand_data('past_data/PUB_Demand_2021.csv')
    print("Loading 2022 demand data...")
    demand_2022 = load_demand_data('past_data/PUB_Demand_2022.csv')
    print("Loading 2023 demand data...")
    demand_2023 = load_demand_data('past_data/PUB_Demand_2023.csv')

    # Prepare training data
    def prepare_training_data(demand_df):
        # Enhanced daily demand calculations
        daily_demand = demand_df.groupby([
            'day_of_year', 'day_of_week', 'month', 
            'is_winter', 'is_summer'
        ]).agg({
            'demand': ['mean', 'max', 'min', 'std'],
            'is_peak_hour': 'mean'  # Percentage of peak hours
        }).reset_index()
        
        # Flatten column names
        daily_demand.columns = ['day_of_year', 'day_of_week', 'month', 
                              'is_winter', 'is_summer', 'demand_mean', 
                              'demand_max', 'demand_min', 'demand_std', 
                              'peak_hour_ratio']
        return daily_demand

    print("Preparing training data...")
    train_2021 = prepare_training_data(demand_2021)
    train_2022 = prepare_training_data(demand_2022)
    train_2023 = prepare_training_data(demand_2023)

    # Combine all training data first
    all_training_data = pd.concat([train_2021, train_2022, train_2023])
    print(f"Total training samples before filtering: {len(all_training_data)}")

    # Prepare prediction data with enhanced features
    print("Preparing prediction data...")
    X_pred = weather_2024.copy()
    X_pred['month'] = X_pred['Date/Time'].dt.month
    X_pred['is_winter'] = X_pred['month'].isin([12, 1, 2])
    X_pred['is_summer'] = X_pred['month'].isin([6, 7, 8])
    
    # Add temperature interactions and derived features
    X_pred['temp_range'] = X_pred['Max Temp (°C)'] - X_pred['Min Temp (°C)']
    X_pred['temp_mean'] = (X_pred['Max Temp (°C)'] + X_pred['Min Temp (°C)']) / 2
    X_pred['extreme_temp'] = (X_pred['Max Temp (°C)'].abs() > 25) | (X_pred['Min Temp (°C)'].abs() > 20)
    
    # Keep relevant columns
    X_pred = X_pred[[
        'day_of_year', 'day_of_week', 'month', 'is_winter', 'is_summer',
        'Max Temp (°C)', 'Min Temp (°C)', 'temp_range', 'temp_mean', 'extreme_temp'
    ]].copy()
    X_pred = X_pred.dropna().sort_values('day_of_year')

    # Create training features with enhanced feature set
    print("Preparing features...")
    X_train = []
    y_train = []

    for day in X_pred['day_of_year'].unique():
        day_weather = X_pred[X_pred['day_of_year'] == day].iloc[0]
        historical_demand = all_training_data[all_training_data['day_of_year'] == day]
        
        if not historical_demand.empty:
            for _, hist_row in historical_demand.iterrows():
                features = {
                    'day_of_year': day,
                    'day_of_week': hist_row['day_of_week'],
                    'month': day_weather['month'],
                    'is_winter': day_weather['is_winter'],
                    'is_summer': day_weather['is_summer'],
                    'Max Temp (°C)': day_weather['Max Temp (°C)'],
                    'Min Temp (°C)': day_weather['Min Temp (°C)'],
                    'temp_range': day_weather['temp_range'],
                    'temp_mean': day_weather['temp_mean'],
                    'extreme_temp': day_weather['extreme_temp']
                }
                X_train.append(features)
                y_train.append(hist_row['demand_mean'])

    X_train = pd.DataFrame(X_train)
    y_train = np.array(y_train)

    print(f"Training with {len(X_train)} samples...")

    # Enhanced preprocessing pipeline with explicit categories
    numeric_features = ['day_of_year', 'Max Temp (°C)', 'Min Temp (°C)', 
                       'temp_range', 'temp_mean']
    categorical_features = ['day_of_week', 'month']
    boolean_features = ['is_winter', 'is_summer', 'extreme_temp']

    # Define explicit categories for day_of_week (0-6) and month (1-12)
    categories = [
        list(range(7)),  # 0-6 for days of week
        list(range(1, 13))  # 1-12 for months
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(
                drop='first', 
                sparse_output=False,
                categories=categories,
                handle_unknown='ignore'  # Ignore any unknown categories
            ), categorical_features),
            ('bool', 'passthrough', boolean_features)
        ]
    )

    # Create both RF and XGBoost pipelines with the same preprocessor
    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ))
    ])

    # For XGBoost, we'll handle cross-validation manually
    xgb_base = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # Evaluate Random Forest model
    print("\nEvaluating Random Forest model...")
    tscv = TimeSeriesSplit(n_splits=5)
    rf_scores = cross_val_score(
        rf_model, 
        X_train, 
        y_train, 
        cv=tscv, 
        scoring='neg_root_mean_squared_error',
        error_score='raise'
    )
    rf_rmse = -rf_scores
    print(f"RF Cross-validation RMSE scores: {rf_rmse}")
    print(f"RF Average RMSE: {rf_rmse.mean():.2f} (+/- {rf_rmse.std() * 2:.2f})")

    # Manual cross-validation for XGBoost
    print("\nEvaluating XGBoost model...")
    xgb_scores = []
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, val_idx in tscv.split(X_train):
        # Split data
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        # Preprocess data
        X_train_processed = preprocessor.fit_transform(X_train_cv)
        X_val_processed = preprocessor.transform(X_val_cv)
        
        # Train and evaluate
        xgb_base.fit(X_train_processed, y_train_cv)
        y_pred = xgb_base.predict(X_val_processed)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
        xgb_scores.append(rmse)
    
    xgb_rmse = np.array(xgb_scores)
    print(f"XGB Cross-validation RMSE scores: {xgb_rmse}")
    print(f"XGB Average RMSE: {xgb_rmse.mean():.2f} (+/- {xgb_rmse.std() * 2:.2f})")

    # Create Neural Network model
    def create_nn_model(input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Manual cross-validation for Neural Network
    print("\nEvaluating Neural Network model...")
    nn_scores = []
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Get the preprocessed feature dimension
    X_sample_processed = preprocessor.fit_transform(X_train.head(1))
    input_dim = X_sample_processed.shape[1]
    
    for train_idx, val_idx in tscv.split(X_train):
        # Split data
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        # Preprocess data
        X_train_processed = preprocessor.fit_transform(X_train_cv)
        X_val_processed = preprocessor.transform(X_val_cv)
        
        # Create and train model
        nn_model = create_nn_model(input_dim)
        nn_model.fit(
            X_train_processed, y_train_cv,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_processed, y_val_cv),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        y_pred = nn_model.predict(X_val_processed, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
        nn_scores.append(rmse)
    
    nn_rmse = np.array(nn_scores)
    print(f"NN Cross-validation RMSE scores: {nn_rmse}")
    print(f"NN Average RMSE: {nn_rmse.mean():.2f} (+/- {nn_rmse.std() * 2:.2f})")

    # Train final models
    print("\nTraining final models...")
    rf_model.fit(X_train, y_train)
    
    # Train final XGBoost model
    X_train_processed = preprocessor.fit_transform(X_train)
    xgb_base.fit(X_train_processed, y_train)
    
    # Get feature names after preprocessing
    feature_names = (numeric_features + 
                    [f"{feat}_{val}" for feat, vals in 
                     zip(categorical_features, preprocessor.named_transformers_['cat'].categories_) 
                     for val in vals[1:]] +
                    boolean_features)
    
    # RF Feature Importance
    rf_importances = rf_model.named_steps['regressor'].feature_importances_
    rf_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_importances
    }).sort_values('importance', ascending=False)
    
    # XGB Feature Importance
    xgb_importances = xgb_base.feature_importances_
    xgb_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_importances
    }).sort_values('importance', ascending=False)
    
    print("\nRandom Forest - Top 10 most important features:")
    print(rf_importance_df.head(10))
    
    print("\nXGBoost - Top 10 most important features:")
    print(xgb_importance_df.head(10))

    # Make predictions
    print("\nMaking predictions...")
    rf_predictions = rf_model.predict(X_pred)
    
    # Process prediction data for XGBoost
    X_pred_processed = preprocessor.transform(X_pred)
    xgb_predictions = xgb_base.predict(X_pred_processed)

    # Create Neural Network model
    def create_nn_model(input_dim):
        model = Sequential([
            Dense(64, activation='relu', input_dim=input_dim),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Manual cross-validation for Neural Network
    print("\nEvaluating Neural Network model...")
    nn_scores = []
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Get the preprocessed feature dimension
    X_sample_processed = preprocessor.fit_transform(X_train.head(1))
    input_dim = X_sample_processed.shape[1]
    
    for train_idx, val_idx in tscv.split(X_train):
        # Split data
        X_train_cv, X_val_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
        
        # Preprocess data
        X_train_processed = preprocessor.fit_transform(X_train_cv)
        X_val_processed = preprocessor.transform(X_val_cv)
        
        # Create and train model
        nn_model = create_nn_model(input_dim)
        nn_model.fit(
            X_train_processed, y_train_cv,
            epochs=100,
            batch_size=32,
            validation_data=(X_val_processed, y_val_cv),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate
        y_pred = nn_model.predict(X_val_processed, verbose=0)
        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred))
        nn_scores.append(rmse)
    
    nn_rmse = np.array(nn_scores)
    print(f"NN Cross-validation RMSE scores: {nn_rmse}")
    print(f"NN Average RMSE: {nn_rmse.mean():.2f} (+/- {nn_rmse.std() * 2:.2f})")

    # Train final Neural Network model
    print("\nTraining final Neural Network model...")
    final_nn_model = create_nn_model(input_dim)
    X_train_processed = preprocessor.fit_transform(X_train)
    final_nn_model.fit(
        X_train_processed, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=0
    )

    # Make predictions with Neural Network
    nn_predictions = final_nn_model.predict(X_pred_processed, verbose=0).flatten()

    # Update ensemble predictions to include Neural Network
    ensemble_predictions = (rf_predictions + xgb_predictions + nn_predictions) / 3

    # Save predictions with additional metadata
    output_data = {
        'dates': X_pred['day_of_year'].tolist(),
        'rf_predictions': rf_predictions.tolist(),
        'xgb_predictions': xgb_predictions.tolist(),
        'nn_predictions': nn_predictions.tolist(),
        'ensemble_predictions': ensemble_predictions.tolist(),
        'model_info': {
            'rf_rmse_mean': float(rf_rmse.mean()),
            'rf_rmse_std': float(rf_rmse.std()),
            'xgb_rmse_mean': float(xgb_rmse.mean()),
            'xgb_rmse_std': float(xgb_rmse.std()),
            'nn_rmse_mean': float(nn_rmse.mean()),
            'nn_rmse_std': float(nn_rmse.std()),
            'rf_top_features': rf_importance_df['feature'].tolist()[:5],
            'xgb_top_features': xgb_importance_df['feature'].tolist()[:5]
        }
    }

    print("Saving predictions...")
    with open('predictions.json', 'w') as f:
        json.dump(output_data, f)
    
    print("Done! Predictions have been saved to predictions.json")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Stack trace:")
    import traceback
    traceback.print_exc() 