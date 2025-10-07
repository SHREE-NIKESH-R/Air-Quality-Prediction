import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("AIR QUALITY PREDICTION MODEL TRAINING")
print("US EPA AQI Standard (Like IQAir)")
print("="*70)

# Load the dataset
print("\nLoading global air quality dataset...")
df = pd.read_csv('global_air_quality_data_10000.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# US EPA AQI Calculation Functions
def calculate_aqi_pm25(pm25):
    """Calculate US AQI for PM2.5"""
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    if pm25 <= 12.0:
        return round(((50 - 0) / (12.0 - 0.0)) * (pm25 - 0.0) + 0)
    elif pm25 <= 35.4:
        return round(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4:
        return round(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4:
        return round(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4:
        return round(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)
    elif pm25 <= 350.4:
        return round(((400 - 301) / (350.4 - 250.5)) * (pm25 - 250.5) + 301)
    else:
        return round(((500 - 401) / (500.4 - 350.5)) * (pm25 - 350.5) + 401)

def calculate_aqi_pm10(pm10):
    """Calculate US AQI for PM10"""
    if pd.isna(pm10) or pm10 < 0:
        return np.nan
    if pm10 <= 54:
        return round(((50 - 0) / (54 - 0)) * (pm10 - 0) + 0)
    elif pm10 <= 154:
        return round(((100 - 51) / (154 - 55)) * (pm10 - 55) + 51)
    elif pm10 <= 254:
        return round(((150 - 101) / (254 - 155)) * (pm10 - 155) + 101)
    elif pm10 <= 354:
        return round(((200 - 151) / (354 - 255)) * (pm10 - 255) + 151)
    elif pm10 <= 424:
        return round(((300 - 201) / (424 - 355)) * (pm10 - 355) + 201)
    elif pm10 <= 504:
        return round(((400 - 301) / (504 - 425)) * (pm10 - 425) + 301)
    else:
        return round(((500 - 401) / (604 - 505)) * (pm10 - 505) + 401)

def calculate_aqi_o3(o3):
    """Calculate US AQI for O3 (8-hour)"""
    if pd.isna(o3) or o3 < 0:
        return np.nan
    if o3 <= 54:
        return round(((50 - 0) / (54 - 0)) * (o3 - 0) + 0)
    elif o3 <= 70:
        return round(((100 - 51) / (70 - 55)) * (o3 - 55) + 51)
    elif o3 <= 85:
        return round(((150 - 101) / (85 - 71)) * (o3 - 71) + 101)
    elif o3 <= 105:
        return round(((200 - 151) / (105 - 86)) * (o3 - 86) + 151)
    elif o3 <= 200:
        return round(((300 - 201) / (200 - 106)) * (o3 - 106) + 201)
    else:
        return 301

# Calculate AQI for each pollutant
print("\n" + "="*70)
print("CALCULATING US EPA AQI...")
print("="*70)

df['AQI_PM25'] = df['PM2.5'].apply(calculate_aqi_pm25)
df['AQI_PM10'] = df['PM10'].apply(calculate_aqi_pm10)
df['AQI_O3'] = df['O3'].apply(calculate_aqi_o3)

# Overall AQI is the maximum of all pollutant AQIs
df['AQI'] = df[['AQI_PM25', 'AQI_PM10', 'AQI_O3']].max(axis=1)

print("✅ AQI calculated using US EPA standard")
print(f"\nAQI Statistics:")
print(f"  Mean AQI: {df['AQI'].mean():.1f}")
print(f"  Min AQI: {df['AQI'].min():.1f}")
print(f"  Max AQI: {df['AQI'].max():.1f}")

# Data preprocessing
print("\n" + "="*70)
print("PREPROCESSING DATA...")
print("="*70)

# Select relevant features
features = ['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']
target = 'AQI'

# Keep only valid data (already clean based on debug)
df_clean = df[features + [target]].copy()

# Remove any negative values or outliers
for col in features:
    df_clean = df_clean[df_clean[col] >= 0]

# Remove any rows with null AQI
df_clean = df_clean.dropna()

print(f"Clean dataset shape: {df_clean.shape}")
print(f"\nDataset statistics:\n{df_clean.describe()}")

# Split features and target
X = df_clean[features]
y = df_clean[target]

print(f"\nFeatures: {features}")
print(f"Target range: {y.min():.1f} to {y.max():.1f}")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]:,}")
print(f"Test set size: {X_test.shape[0]:,}")

# Model training and comparison
print("\n" + "="*70)
print("TRAINING MODELS...")
print("="*70)

models = {
    'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
    'XGBoost': xgb.XGBRegressor(random_state=42, objective='reg:squarederror', n_estimators=100)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'rmse': test_rmse,
        'mae': test_mae
    }
    
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['test_r2'])
best_model = results[best_model_name]['model']

print("\n" + "="*70)
print(f"BEST MODEL: {best_model_name}")
print("="*70)
print(f"Test R² Score: {results[best_model_name]['test_r2']:.4f}")
print(f"RMSE: {results[best_model_name]['rmse']:.4f}")
print(f"MAE: {results[best_model_name]['mae']:.4f}")

# Hyperparameter tuning
print("\n" + "="*70)
print("HYPERPARAMETER TUNING...")
print("="*70)

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5]
    }
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
else:
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }

grid_search = GridSearchCV(
    best_model, param_grid, cv=5,
    scoring='r2', n_jobs=-1, verbose=1
)

print(f"Running Grid Search on {best_model_name}...")
grid_search.fit(X_train_scaled, y_train)

# Best model after tuning
final_model = grid_search.best_estimator_
y_pred_final = final_model.predict(X_test_scaled)

final_r2 = r2_score(y_test, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_mae = mean_absolute_error(y_test, y_pred_final)

print("\n" + "="*70)
print("FINAL MODEL PERFORMANCE")
print("="*70)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"R² Score: {final_r2:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"MAE: {final_mae:.4f}")
print(f"Accuracy: {final_r2*100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)
print(feature_importance)

# Save model and scaler
print("\n" + "="*70)
print("SAVING MODEL...")
print("="*70)

with open('aqi_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save metadata
metadata = {
    'aqi_standard': 'US EPA AQI',
    'features': features,
    'model_type': best_model_name,
    'accuracy_r2': final_r2,
    'rmse': final_rmse,
    'mae': final_mae,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'date_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("✅ Model saved as 'aqi_model.pkl'")
print("✅ Scaler saved as 'scaler.pkl'")
print("✅ Metadata saved as 'model_metadata.pkl'")

print(f"\n{'='*70}")
print("MODEL SUMMARY")
print("="*70)
print(f"📊 AQI Standard: US EPA (Same as IQAir)")
print(f"📈 Training Data: {len(X_train):,} samples")
print(f"🎯 Test Data: {len(X_test):,} samples")
print(f"🏆 Best Model: {best_model_name}")
print(f"📐 Final R² Score: {final_r2:.4f} ({final_r2*100:.2f}% accuracy)")
print(f"📉 RMSE: {final_rmse:.2f}")
print(f"📉 MAE: {final_mae:.2f}")
print("\n🎉 MODEL TRAINING COMPLETE!")
print("="*70)
print("\nNext: Run 'streamlit run app.py' to start the dashboard")