import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from itertools import combinations
from sklearn.model_selection import GridSearchCV
import joblib

# Load dataset
real_estate_data = pd.read_excel('')

# Check if 'TRUE_SITE_CITY' exists in the DataFrame
if 'TRUE_SITE_CITY' in real_estate_data.columns:
    # Dictionary to store models for each city
    city_models = {}

    # Iterate over each city
    for city in real_estate_data['TRUE_SITE_CITY'].unique():
        # Filter data for the current city
        city_data = real_estate_data[real_estate_data['TRUE_SITE_CITY'] == city]

        # Check if target variable exists
        if 'Average Flip Days' not in city_data.columns:
            print(f"Error: Target variable 'Average Flip Days' not found for city {city}. Skipping.")
            continue  # Skip to the next city if target variable is missing

        # Select numerical columns for the current city
        numerical_columns = city_data.select_dtypes(include=['int64', 'float64']).columns

        # Initialize variables to store the best combination of features and its performance
        best_features = []
        best_performance = float('inf')  # Initialize with a large value for MSE
        best_params = None  # Initialize to store best hyperparameters

        # Consider early stopping criteria (e.g., maximum number of combinations to try)
        max_combinations = 100

        # Iterate through a limited number of possible feature combinations
        for k in range(1, len(numerical_columns) + 1):
            if len(best_features) > 0 and len(combinations(numerical_columns, k)) > max_combinations - len(best_features):
                break  # Early stopping if enough combinations have been explored
            for feature_combination in combinations(numerical_columns, k):
                # Prepare features for modeling
                city_features = city_data[list(feature_combination)]
                imputer = SimpleImputer(strategy='mean')
                city_features_imputed = imputer.fit_transform(city_features)

                # Define hyperparameter grid for GridSearchCV (optional)
                param_grid = {
                    'linearregression__normalize': [True, False],
                }

                # Use GridSearchCV to find best model and hyperparameters (optional)
                model = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=5)
                model.fit(city_features_imputed, city_data['Average Flip Days'])
                y_pred = model.predict(city_features_imputed)
                mse = mean_squared_error(city_data['Average Flip Days'], y_pred)

                # Update the best combination and hyperparameters if the current one performs better
                if mse < best_performance:
                    best_features = feature_combination
                    best_performance = mse
                    best_params = model.best_params_  # Store best hyperparameters

        # Train a final model using the best combination of features for the current city
        city_features = city_data[list(best_features)]
        imputer = SimpleImputer(strategy='mean')
        city_features_imputed = imputer.fit_transform(city_features)

        # Train the model using potentially optimized hyperparameters (from GridSearchCV)
        if best_params is not None:
            final_model = LinearRegression(**best_params)  # Unpack best hyperparameters
        else:
            final_model = LinearRegression()
        final_model.fit(city_features_imputed, city_data['Average Flip Days'])

        # Calculate R-squared for the final model
        y_pred_final = final_model.predict(city_features_imputed)
        r2_final = r2_score(city_data['Average Flip Days'], y_pred_final)

        # Print the results for the current city
        print(f"City: {city}")
        print(f"Best Features: {best_features}")
        print(f"Best Performance (MSE): {best_performance}")
        print(f"Best Performance (R-squared): {r2_final}")
        print(f"Best Hyperparameters: {best_params}")

        # Save the final model for the current city
        joblib.dump(final_model, f'{city}_model.pkl')
        city_models[city] = final_model

    # Save the dictionary containing models for each city
    joblib.dump(city_models, 'city_models.pkl')
else:
    print("Error: Column 'TRUE_SITE_CITY' not found in the DataFrame.")

