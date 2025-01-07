import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def prepare_training_data(demand_df):
    # Base features for grouping
    group_cols = [
        'day_of_year', 'day_of_week', 'month', 'week_of_year',
        'is_winter', 'is_summer', 'is_spring', 'is_fall'
    ]
    
    # Add weather columns if they exist
    weather_cols = [
        'Max Temp (°C)', 'Min Temp (°C)', 'temp_mean', 'temp_range',
        'temp_mean_3d_avg', 'temp_range_3d_avg', 'max_temp_3d_avg', 'min_temp_3d_avg',
        'temp_mean_7d_avg', 'temp_range_7d_avg', 'max_temp_7d_avg', 'min_temp_7d_avg',
        'temp_mean_change', 'temp_range_change'
    ]
    
    existing_weather_cols = [col for col in weather_cols if col in demand_df.columns]
    if existing_weather_cols:
        group_cols.extend(existing_weather_cols)
    
    # Group by all available features
    daily_demand = demand_df.groupby(group_cols).agg({
        'demand': ['mean', 'max', 'min', 'std'],
        'is_peak_hour': 'mean'
    }).reset_index()
    
    # Flatten column names
    daily_demand.columns = (
        group_cols +
        ['demand_mean', 'demand_max', 'demand_min', 'demand_std', 'peak_hour_ratio']
    )
    
    # Add derived temperature features if weather data exists
    if 'temp_mean' in daily_demand.columns:
        daily_demand['temp_mean_squared'] = daily_demand['temp_mean'] ** 2
        daily_demand['temp_above_25'] = np.maximum(0, daily_demand['temp_mean'] - 25)
        daily_demand['temp_below_5'] = np.maximum(0, 5 - daily_demand['temp_mean'])
        daily_demand['summer_temp'] = daily_demand['temp_mean'] * daily_demand['is_summer']
        daily_demand['summer_temp_squared'] = daily_demand['temp_mean_squared'] * daily_demand['is_summer']
        daily_demand['winter_temp'] = daily_demand['temp_mean'] * daily_demand['is_winter']
    
    return daily_demand

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
    
    # Enhanced time-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_winter'] = df['month'].isin([12, 1, 2])
    df['is_summer'] = df['month'].isin([6, 7, 8])
    df['is_spring'] = df['month'].isin([3, 4, 5])
    df['is_fall'] = df['month'].isin([9, 10, 11])
    df['is_peak_hour'] = df['hour'].between(9, 17)
    
    # Calculate daily stats
    daily_stats = df.groupby(df['date'].dt.date).agg({
        'demand': ['mean', 'max', 'min', 'std']
    }).reset_index()
    daily_stats.columns = ['date', 'daily_mean_demand', 'daily_max_demand', 
                          'daily_min_demand', 'daily_demand_std']
    
    df = df.merge(daily_stats, left_on=df['date'].dt.date, right_on='date', 
                 suffixes=('', '_daily'))
    
    return df

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
    
    # Enhanced weather features
    weather_df['temp_mean'] = (weather_df['Max Temp (°C)'] + weather_df['Min Temp (°C)']) / 2
    weather_df['temp_range'] = weather_df['Max Temp (°C)'] - weather_df['Min Temp (°C)']
    
    # Add rolling temperature features (3-day and 7-day)
    weather_df = weather_df.sort_values('Date/Time')
    for window in [3, 7]:
        weather_df[f'temp_mean_{window}d_avg'] = weather_df['temp_mean'].rolling(window=window).mean()
        weather_df[f'temp_range_{window}d_avg'] = weather_df['temp_range'].rolling(window=window).mean()
        weather_df[f'max_temp_{window}d_avg'] = weather_df['Max Temp (°C)'].rolling(window=window).mean()
        weather_df[f'min_temp_{window}d_avg'] = weather_df['Min Temp (°C)'].rolling(window=window).mean()
    
    # Temperature change features
    weather_df['temp_mean_change'] = weather_df['temp_mean'].diff()
    weather_df['temp_range_change'] = weather_df['temp_range'].diff()
    
    weather_df['day_of_year'] = weather_df['Date/Time'].dt.dayofyear
    weather_df['day_of_week'] = weather_df['Date/Time'].dt.dayofweek
    weather_df['month'] = weather_df['Date/Time'].dt.month
    weather_df['week_of_year'] = weather_df['Date/Time'].dt.isocalendar().week
    
    return weather_df

try:
    # Load 2024 weather data for predictions
    print("Loading 2024 weather data...")
    weather_files_2024 = [f'2024_weather_data/en_climate_daily_ON_6158359_2024_P1D_{str(i).zfill(2)}.csv' for i in range(1, 9)]
    weather_2024 = load_weather_data(weather_files_2024)

    # Load historical weather data (2018-2023)
    print("\nLoading historical weather data...")
    weather_data = {}
    for year in range(2018, 2024):
        print(f"Loading {year} weather data...")
        weather_files = [f'{year}_weather_data/en_climate_daily_ON_6158359_{year}_P1D_{str(i).zfill(2)}.csv' for i in range(1, 13)]
        try:
            weather_data[year] = load_weather_data(weather_files)
            print(f"Loaded {len(weather_data[year])} days of weather data for {year}")
        except FileNotFoundError as e:
            print(f"Warning: Some weather data files for {year} not found")

    # Load historical demand data
    print("\nLoading historical demand data...")
    demand_data = {}
    for year in range(2018, 2024):
        print(f"Loading {year} demand data...")
        try:
            demand_data[year] = load_demand_data(f'past_data/PUB_Demand_{year}.csv')
        except FileNotFoundError:
            print(f"Warning: Data file for {year} not found")

    # Prepare training data
    print("\nPreparing training data...")
    training_sets = []
    for year, data in demand_data.items():
        if year in weather_data:
            # Merge demand data with corresponding weather data
            data = data.merge(
                weather_data[year][['Date/Time', 'Max Temp (°C)', 'Min Temp (°C)', 'temp_mean', 'temp_range',
                                  'temp_mean_3d_avg', 'temp_range_3d_avg', 'max_temp_3d_avg', 'min_temp_3d_avg',
                                  'temp_mean_7d_avg', 'temp_range_7d_avg', 'max_temp_7d_avg', 'min_temp_7d_avg',
                                  'temp_mean_change', 'temp_range_change']],
                left_on=data['date'].dt.date,
                right_on=weather_data[year]['Date/Time'].dt.date,
                how='left'
            )
        training_sets.append(prepare_training_data(data))
        print(f"Added {year} data with {len(training_sets[-1])} samples")

    # Combine all training data
    all_training_data = pd.concat(training_sets)
    print(f"\nTotal training samples before filtering: {len(all_training_data)}")

    # Prepare prediction data with enhanced features
    print("Preparing prediction data...")
    X_pred = weather_2024.copy()
    X_pred['month'] = X_pred['Date/Time'].dt.month
    X_pred['is_winter'] = X_pred['month'].isin([12, 1, 2])
    X_pred['is_summer'] = X_pred['month'].isin([6, 7, 8])
    X_pred['is_spring'] = X_pred['month'].isin([3, 4, 5])
    X_pred['is_fall'] = X_pred['month'].isin([9, 10, 11])
    
    # Add non-linear temperature features
    X_pred['temp_mean_squared'] = X_pred['temp_mean'] ** 2
    X_pred['temp_above_25'] = np.maximum(0, X_pred['temp_mean'] - 25)  # Cooling threshold
    X_pred['temp_below_5'] = np.maximum(0, 5 - X_pred['temp_mean'])    # Heating threshold
    
    # Add interaction terms
    X_pred['summer_temp'] = X_pred['temp_mean'] * X_pred['is_summer']
    X_pred['summer_temp_squared'] = X_pred['temp_mean_squared'] * X_pred['is_summer']
    X_pred['winter_temp'] = X_pred['temp_mean'] * X_pred['is_winter']
    
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
                    'week_of_year': day_weather['week_of_year'],
                    'is_winter': day_weather['is_winter'],
                    'is_summer': day_weather['is_summer'],
                    'is_spring': day_weather['is_spring'],
                    'is_fall': day_weather['is_fall'],
                    'Max Temp (°C)': day_weather['Max Temp (°C)'],
                    'Min Temp (°C)': day_weather['Min Temp (°C)'],
                    'temp_mean': day_weather['temp_mean'],
                    'temp_range': day_weather['temp_range'],
                    'temp_mean_3d_avg': day_weather['temp_mean_3d_avg'],
                    'temp_range_3d_avg': day_weather['temp_range_3d_avg'],
                    'max_temp_3d_avg': day_weather['max_temp_3d_avg'],
                    'min_temp_3d_avg': day_weather['min_temp_3d_avg'],
                    'temp_mean_7d_avg': day_weather['temp_mean_7d_avg'],
                    'temp_range_7d_avg': day_weather['temp_range_7d_avg'],
                    'max_temp_7d_avg': day_weather['max_temp_7d_avg'],
                    'min_temp_7d_avg': day_weather['min_temp_7d_avg'],
                    'temp_mean_change': day_weather['temp_mean_change'],
                    'temp_range_change': day_weather['temp_range_change'],
                    'temp_mean_squared': day_weather['temp_mean'] ** 2,
                    'temp_above_25': max(0, day_weather['temp_mean'] - 25),
                    'temp_below_5': max(0, 5 - day_weather['temp_mean']),
                    'summer_temp': day_weather['temp_mean'] * day_weather['is_summer'],
                    'summer_temp_squared': (day_weather['temp_mean'] ** 2) * day_weather['is_summer'],
                    'winter_temp': day_weather['temp_mean'] * day_weather['is_winter']
                }
                X_train.append(features)
                y_train.append(hist_row['demand_mean'])

    X_train = pd.DataFrame(X_train)
    y_train = np.array(y_train)

    print(f"Training with {len(X_train)} samples...")

    # Enhanced preprocessing pipeline with new features
    numeric_features = ['day_of_year', 'week_of_year', 'Max Temp (°C)', 'Min Temp (°C)', 
                       'temp_mean', 'temp_range', 
                       'temp_mean_3d_avg', 'temp_range_3d_avg',
                       'max_temp_3d_avg', 'min_temp_3d_avg',
                       'temp_mean_7d_avg', 'temp_range_7d_avg',
                       'max_temp_7d_avg', 'min_temp_7d_avg',
                       'temp_mean_change', 'temp_range_change',
                       'temp_mean_squared', 'temp_above_25', 'temp_below_5',
                       'summer_temp', 'summer_temp_squared', 'winter_temp']
    categorical_features = ['day_of_week', 'month']
    boolean_features = ['is_winter', 'is_summer', 'is_spring', 'is_fall']

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
                handle_unknown='ignore'
            ), categorical_features),
            ('bool', 'passthrough', boolean_features)
        ]
    )

    # First stage: Initial model to determine feature importance
    print("\nStage 1: Training initial model for feature selection...")
    initial_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    initial_rf.fit(X_train, y_train)

    # Get feature importance from initial model
    feature_names = (numeric_features + 
                    [f"{feat}_{val}" for feat, vals in 
                     zip(categorical_features, initial_rf.named_steps['preprocessor']
                         .named_transformers_['cat'].categories_) 
                     for val in vals[1:]] +
                    boolean_features)
    
    importances = initial_rf.named_steps['regressor'].feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nInitial feature importance ranking:")
    print(importance_df)

    # Select top features (those with importance > mean importance)
    importance_threshold = importance_df['importance'].mean()
    selected_features = importance_df[importance_df['importance'] > importance_threshold]
    print(f"\nSelected {len(selected_features)} features with importance > {importance_threshold:.4f}")
    print("Selected features:")
    print(selected_features)

    # Create new feature lists based on selection
    selected_numeric = [f for f in numeric_features 
                       if f in selected_features['feature'].values]
    selected_categorical = [f for f in categorical_features 
                          if any(f in feat for feat in selected_features['feature'].values)]
    selected_boolean = [f for f in boolean_features 
                       if f in selected_features['feature'].values]

    # Create new preprocessor with selected features
    selected_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), selected_numeric),
            ('cat', OneHotEncoder(
                drop='first', 
                sparse_output=False,
                categories=[categories[i] for i, feat in enumerate(categorical_features) 
                          if any(feat in f for f in selected_features['feature'].values)],
                handle_unknown='ignore'
            ), selected_categorical),
            ('bool', 'passthrough', selected_boolean)
        ]
    )

    # Define expanded parameter grid for Random Forest
    param_grid = {
        'regressor__n_estimators': [300, 400, 500],
        'regressor__max_depth': [15, 20, 25, 30],
        'regressor__min_samples_split': [2, 5, 8],
        'regressor__min_samples_leaf': [2, 4, 6],
        'regressor__max_features': ['sqrt', 'log2'],
        'regressor__bootstrap': [True],  # Simplified parameter grid
        'regressor__warm_start': [False]  # Simplified parameter grid
    }

    # Create pipeline with selected features
    rf_pipeline = Pipeline([
        ('preprocessor', selected_preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # Perform grid search with time series cross-validation
    print("\nStage 2: Performing grid search with selected features...")
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=tscv,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)
    print(f"\nBest RMSE: {-grid_search.best_score_:.2f}")

    # Get cross-validation scores for the best model
    best_rf_scores = cross_val_score(
        grid_search.best_estimator_,
        X_train,
        y_train,
        cv=tscv,
        scoring='neg_root_mean_squared_error'
    )
    best_rf_rmse = -best_rf_scores

    print(f"\nCross-validation RMSE scores: {best_rf_rmse}")
    print(f"Average RMSE: {best_rf_rmse.mean():.2f} (+/- {best_rf_rmse.std() * 2:.2f})")

    # Get final feature importance
    final_feature_names = (selected_numeric + 
                         [f"{feat}_{val}" for feat, vals in 
                          zip(selected_categorical, grid_search.best_estimator_.named_steps['preprocessor']
                              .named_transformers_['cat'].categories_) 
                          for val in vals[1:]] +
                         selected_boolean)
    
    final_importances = grid_search.best_estimator_.named_steps['regressor'].feature_importances_
    final_importance_df = pd.DataFrame({
        'feature': final_feature_names,
        'importance': final_importances
    }).sort_values('importance', ascending=False)
    
    print("\nFinal feature importance ranking:")
    print(final_importance_df)

    # Make predictions with the best model
    print("\nMaking predictions...")
    predictions = grid_search.best_estimator_.predict(X_pred)

    # Save predictions and model information
    output_data = {
        'dates': X_pred['day_of_year'].tolist(),
        'rf_predictions': predictions.tolist(),
        'model_info': {
            'rmse_mean': float(best_rf_rmse.mean()),
            'rmse_std': float(best_rf_rmse.std()),
            'best_params': grid_search.best_params_,
            'selected_features': final_feature_names,
            'feature_importance': {
                'features': final_importance_df['feature'].tolist(),
                'importance': final_importance_df['importance'].tolist()
            }
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