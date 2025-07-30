import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Load data
def load_data(file_path=r"C:\Users\Kritin\Downloads\driving_data.csv"):
    df = pd.read_csv(file_path)
    
    # Ensure UTC time is properly formatted
    if 'UTC Time' in df.columns:
        df = df.rename(columns={'UTC Time': 'utc_time'})
    
    # Convert column names to lowercase and remove spaces
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    # Convert string columns to numeric if needed
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    # Add a distance column if not present
    if 'dist' not in df.columns:
        df['dist'] = 0.0
        # Calculate distance from speed
        if 'speed' in df.columns and 'utc_time' in df.columns:
            for i in range(1, len(df)):
                time_diff = df.loc[i, 'utc_time'] - df.loc[i-1, 'utc_time']
                avg_speed = (df.loc[i, 'speed'] + df.loc[i-1, 'speed']) / 2
                df.loc[i, 'dist'] = avg_speed * time_diff / 3600  # Convert to meters
    
    # Add acceleration column if not present
    if 'acceleration' not in df.columns and 'longitudinal_acc' in df.columns:
        df['acceleration'] = df['longitudinal_acc']
    
    return df

def compute_retro_metrics(df_window, params):
    """Compute multiple retrofitting metrics from a window of driving data"""
    
    # Extract parameters
    battery_cost = params.get('battery_cost', 20000)
    fuel_eff_km = params.get('fuel_eff_km', 0.03)
    fuel_price = params.get('fuel_price', 100.0)
    elec_eff_km = params.get('elec_eff_km', 0.015)
    elec_price = params.get('elec_price', 8.0)
    co2_factor = params.get('co2_factor', 0.07)
    co2_price = params.get('co2_price', 0.05)
    
    # Calculate time difference
    times = df_window['utc_time'].values
    total_seconds = times[-1] - times[0]
    total_hours = total_seconds / 3600.0
    
    # Skip calculations if time window is too small
    if total_hours < 0.001:  # Less than 3.6 seconds
        return None
    
    # Calculate distance
    total_km = df_window['dist'].sum() / 1000.0
    
    # If no distance, try to estimate from speed
    if total_km < 0.0001:
        avg_speed = df_window['speed'].mean()  # in km/h
        total_km = avg_speed * total_hours
    
    # Skip if distance is still too small
    if total_km < 0.01:
        return None
    
    # Calculate annual distance
    hours_per_year = 2400
    annual_km = total_km * (hours_per_year / total_hours)
    
    # Driving pattern metrics
    avg_speed = df_window['speed'].mean()
    if 'longitudinal_acc' in df_window and 'lateral_acc' in df_window:
        driving_intensity = df_window[['longitudinal_acc', 'lateral_acc']].abs().mean().mean()
    else:
        driving_intensity = df_window['acceleration'].abs().mean() if 'acceleration' in df_window else 0.1
    
    acceleration_var = df_window['acceleration'].var() if 'acceleration' in df_window else 0.01
    
    # Economic calculations
    fuel_sav_per_km = fuel_eff_km * fuel_price
    elec_cost_per_km = elec_eff_km * elec_price
    co2_sav_per_km = co2_factor * co2_price
    
    net_sav_per_km = fuel_sav_per_km - elec_cost_per_km + co2_sav_per_km
    annual_savings = annual_km * net_sav_per_km
    
    # Break-even years
    retro_age = battery_cost / annual_savings if annual_savings > 0 else np.inf
    
    # Cap retro_age to prevent extreme values
    if np.isinf(retro_age) or retro_age > 30:
        retro_age = 30.0
    
    return {
        'retro_age': retro_age,
        'annual_savings': annual_savings,
        'driving_intensity': driving_intensity,
        'avg_speed': avg_speed,
        'acceleration_var': acceleration_var,
        'annual_km': annual_km
    }

def generate_features(df, window_size=50, params=None):
    """Generate features for ML model based on driving patterns"""
    if params is None:
        params = {
            'battery_cost': 20000,
            'fuel_eff_km': 0.03,
            'fuel_price': 100.0,
            'elec_eff_km': 0.015,
            'elec_price': 8.0,
            'co2_factor': 0.07,
            'co2_price': 0.05
        }
    
    # Initialize DataFrame to store features
    features_df = pd.DataFrame()
    
    # We'll use a step size to reduce computation load
    step_size = max(1, window_size // 10)
    
    # Apply sliding window for feature extraction
    for i in range(0, len(df) - window_size, step_size):
        window = df.iloc[i:i+window_size]
        metrics = compute_retro_metrics(window, params)
        
        if metrics is not None:
            # Add timestamp or index as identifier
            metrics['window_start'] = i
            metrics['window_end'] = i + window_size
            
            # Add to features dataframe
            features_df = pd.concat([features_df, pd.DataFrame([metrics])], ignore_index=True)
    
    # Check if we have any features
    if features_df.empty:
        return pd.DataFrame()
    
    # Handle infinite values
    features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return features_df

def train_ml_models(features_df):
    """Train ML models to predict optimal retrofitting age"""
    
    # Check if there's enough data for training
    if len(features_df) < 10:  # Arbitrary threshold, adjust as needed
        st.error("Not enough data points for training. Try reducing the window size or check your data.")
        return None, None, None
    
    # Prepare features and target
    X = features_df.drop(['retro_age', 'window_start', 'window_end'], axis=1, errors='ignore')
    y = features_df['retro_age']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
    return results, scaler, X.columns

def find_optimal_retrofitting_policy(df, results, scaler, feature_cols):
    """Find optimal retrofitting policy based on ML predictions"""
    
    # Get best model
    best_model_name = min(results, key=lambda x: results[x]['mse'])
    best_model = results[best_model_name]['model']
    
    # Drop window columns if they exist
    feature_columns = [col for col in feature_cols if col not in ['window_start', 'window_end']]
    
    # Create feature vector for each data point
    X_full = df[feature_columns].values
    X_full_scaled = scaler.transform(X_full)
    
    # Make predictions
    predicted_ages = best_model.predict(X_full_scaled)
    
    # Add predictions to dataframe
    df['predicted_retro_age'] = predicted_ages
    
    # Find optimal retrofitting points
    # (points where predicted age is lowest represent good retrofitting opportunities)
    threshold = df['predicted_retro_age'].quantile(0.1)
    optimal_points = df[df['predicted_retro_age'] <= threshold]
    
    return df, optimal_points, best_model_name

def main():
    # Set up Streamlit app
    st.title("Optimal EV Retrofitting Age Predictor")
    
    # Load data
    df = load_data(r"C:\Users\Kritin\Downloads\driving_data.csv")
    
    # Display data sample
    st.subheader("Sample of Driving Data")
    st.write(df.head())
    st.write(f"Total data points: {len(df)}")
    
    # Parameter inputs
    st.sidebar.header("Retrofitting Parameters")
    params = {
        'battery_cost': st.sidebar.slider("Battery Cost (₹)", 10000, 50000, 20000),
        'fuel_eff_km': st.sidebar.slider("Fuel Efficiency (L/km)", 0.01, 0.10, 0.03, 0.01),
        'fuel_price': st.sidebar.slider("Fuel Price (₹/L)", 50.0, 150.0, 100.0, 5.0),
        'elec_eff_km': st.sidebar.slider("Electricity Efficiency (kWh/km)", 0.005, 0.025, 0.015, 0.001),
        'elec_price': st.sidebar.slider("Electricity Price (₹/kWh)", 4.0, 12.0, 8.0, 0.5),
        'co2_factor': st.sidebar.slider("CO₂ Emissions (kg/km)", 0.01, 0.15, 0.07, 0.01),
        'co2_price': st.sidebar.slider("CO₂ Price (₹/kg)", 0.0, 0.5, 0.05, 0.01)
    }
    
    window_size = st.sidebar.slider("Analysis Window Size", 20, 100, 50)
    
    # Generate features
    with st.spinner("Generating features..."):
        features_df = generate_features(df, window_size, params)
        
        # Check if features were generated
        if features_df.empty:
            st.error("No features were generated. Try adjusting the window size or check your data.")
            return
        
        st.success(f"Generated {len(features_df)} feature rows.")
        st.write("Sample of generated features:")
        st.write(features_df.head())
    
    # Train ML models
    with st.spinner("Training ML models..."):
        results, scaler, feature_cols = train_ml_models(features_df)
        
        # Check if training was successful
        if results is None:
            return
    
    # Find optimal retrofitting policy
    df_with_predictions, optimal_points, best_model = find_optimal_retrofitting_policy(
        features_df, results, scaler, feature_cols)
    
    # Display results
    st.subheader(f"Best Model: {best_model}")
    st.write(f"MSE: {results[best_model]['mse']:.4f}, R²: {results[best_model]['r2']:.4f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': results[best_model]['feature_importance'].keys(),
        'Importance': results[best_model]['feature_importance'].values()
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'))
    
    # Plot retrofitting age over time
    st.subheader("Predicted Retrofitting Age Over Time")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(df_with_predictions)), df_with_predictions['predicted_retro_age'], 
            label='Predicted Age')
    ax.plot(range(len(df_with_predictions)), df_with_predictions['retro_age'], 
            alpha=0.5, label='Original Calculation')
    
    # Highlight optimal retrofitting points
    if not optimal_points.empty:
        # Get indices of optimal points
        optimal_indices = optimal_points.index.tolist()
        ax.scatter(optimal_indices, 
                  optimal_points['predicted_retro_age'], 
                  color='red', s=50, label='Optimal Retrofitting Points')
    
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Years to Break Even')
    ax.set_title('Retrofitting Age Prediction')
    ax.legend()
    st.pyplot(fig)
    
    # Show optimal retrofitting opportunities
    st.subheader("Optimal Retrofitting Opportunities")
    st.write(f"Found {len(optimal_points)} optimal retrofitting points out of {len(df_with_predictions)} total points.")
    
    if not optimal_points.empty:
        st.write("Top 5 optimal retrofitting opportunities:")
        st.dataframe(optimal_points.sort_values('predicted_retro_age').head(5))
    
    # Analyze factors affecting retrofitting age
    st.subheader("Factors Affecting Retrofitting Age")
    
    # Create correlation matrix
    corr_matrix = df_with_predictions.corr()
    corr_with_retro = corr_matrix['predicted_retro_age'].sort_values(ascending=True)
    
    # Display correlations
    st.write("Correlation with retrofitting age:")
    st.write(corr_with_retro)
    
    # Plot a few key relationships
    st.subheader("Key Relationships")
    
    # Find the most important feature
    top_feature = importance_df.iloc[0]['Feature']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df_with_predictions[top_feature], df_with_predictions['predicted_retro_age'], alpha=0.5)
    ax.set_xlabel(top_feature)
    ax.set_ylabel('Predicted Retrofitting Age (years)')
    ax.set_title(f'Relationship between {top_feature} and Retrofitting Age')
    st.pyplot(fig)
    
    # Provide recommendations
    st.subheader("Recommendations")
    
    avg_optimal_age = optimal_points['predicted_retro_age'].mean()
    
    st.write(f"Based on the analysis, the optimal retrofitting age is approximately {avg_optimal_age:.2f} years.")
    st.write("Key factors that influence the optimal retrofitting timing:")
    
    for feature, importance in list(importance_df.itertuples(index=False, name=None))[:3]:
        st.write(f"- {feature}: {importance:.2f} importance")
    
    # Summary metrics
    st.subheader("Summary Metrics")
    
    avg_retro_age = df_with_predictions['predicted_retro_age'].mean()
    min_retro_age = df_with_predictions['predicted_retro_age'].min()
    max_retro_age = df_with_predictions['predicted_retro_age'].max()
    
    st.write(f"Average retrofitting age: {avg_retro_age:.2f} years")
    st.write(f"Minimum retrofitting age: {min_retro_age:.2f} years")
    st.write(f"Maximum retrofitting age: {max_retro_age:.2f} years")

if __name__ == "__main__":
    main()