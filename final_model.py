import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split, TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.stats import randint, uniform


def add_cols(df):
    # Sort by player and season
    df = df.sort_values(by=["player_id", "season"])

    # Add prev season stats
    df["prev_season_fantasy_points_ppr"] = df.groupby("player_id")["season_fantasy_points_ppr"].shift(1)
    df["prev_games_played_season"] = df.groupby("player_id")["games_played_season"].shift(1)

    # RB's 
    df["prev_season_rushing_yards"] = df.groupby("player_id")["season_rushing_yards"].shift(1)
    df["prev_season_rush_touchdown"] = df.groupby("player_id")["season_rush_touchdown"].shift(1)
    df["prev_season_rush_attempts"] = df.groupby("player_id")["season_rush_attempts"].shift(1)
    df["prev_season_rush_attempts"] = df.groupby("player_id")["season_rush_attempts"].shift(1)

     # 3-season rolling averages for RB's -> Injury risk
    df["prev3_fantasy_points_ppr"] = df.groupby("player_id")["season_fantasy_points_ppr"].shift(1)\
                                       .rolling(window=3, min_periods=1).mean()
    df["prev3_rushing_yards"] = df.groupby("player_id")["season_rushing_yards"].shift(1)\
                                   .rolling(window=3, min_periods=1).mean()
    df["prev3_rush_touchdowns"] = df.groupby("player_id")["season_rush_touchdown"].shift(1)\
                                      .rolling(window=3, min_periods=1).mean()
    df["prev3_rush_attempts"] = df.groupby("player_id")["season_rush_attempts"].shift(1)\
                                    .rolling(window=3, min_periods=1).mean()

    # WR's
    df["prev2_season_receiving_yards"] = df.groupby("player_id")["season_receiving_yards"].shift(1)\
                                      .rolling(window=2, min_periods=1).mean()
    df["prev2_season_receiving_touchdown"] = df.groupby("player_id")["season_receiving_touchdown"].shift(1)\
                                      .rolling(window=2, min_periods=1).mean()
    df["prev_yards_after_catch"] = df.groupby("player_id")["yards_after_catch"].shift(1)
    df["prev_receiving_air_yards"] = df.groupby("player_id")["receiving_air_yards"].shift(1)
    df["prev_targets"] = df.groupby("player_id")["targets"].shift(1)

    # Change year over year
    df['delta_fantasy_points'] = df.groupby('player_id')['season_fantasy_points_ppr'].diff().clip(-100, 100)

    return df

def feature_engineering(df):
    # Sort by player and season
    df = df.sort_values(by=["player_id", "season"])

    df = add_cols(df)

    # Add column for predicted stats
    df["fantasy_points_ppr_next"] = df.groupby("player_id")["season_fantasy_points_ppr"].shift(-1)

    # Fill missing rolling averages with the overall mean -> Short primes
    rolling_cols = ["prev3_fantasy_points_ppr", "prev3_rushing_yards", "prev3_rush_touchdowns", 
                    "prev3_rush_attempts", "prev2_season_receiving_yards", "prev2_season_receiving_touchdown", 
                    "prev2_season_receiving_yards", "prev_yards_after_catch"]
    for col in rolling_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Drop rows where previous season stats are missing (first season for each player)
    df = df.dropna(subset=[
        "prev_season_fantasy_points_ppr", 
        "prev_season_rushing_yards", 
        "prev_season_rush_touchdown", 
        "prev_season_rush_attempts",
        "prev_yards_after_catch",
        "prev_receiving_air_yards",
        "prev_targets"
        ])

    return df

# Helper function to perform all of our data preprocessing in one place.
# This will be used throughout the project to ensure consistency in our data preprocessing.
# We want to ensure we use the same preprocessing steps when training and deploying the model.
def transform_data(X_train, X_test):
    # Impute missing values using SimpleImputer using "mean"
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features using Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled


def main():
    # Read data for QB's, RB's, and WR's for regular season stats
    df = (pd.read_csv("data/yearly_player_stats_offense.csv").query("season_type == 'REG' and position in ['RB', 'WR', 'TE']"))
    # One-hot encode the positions
    df = pd.get_dummies(df, columns=['position'])

    df = feature_engineering(df)

    columns_to_use = [
        "age", "draft_ovr", "years_exp", "prev_season_fantasy_points_ppr", 
        "prev_season_rushing_yards", "prev_season_rush_touchdown", "prev_season_rush_attempts", 
        "prev_games_played_season", "prev3_fantasy_points_ppr", "prev3_rushing_yards", "prev3_rush_touchdowns", 
        "prev3_rush_attempts", "prev2_season_receiving_yards", "prev2_season_receiving_touchdown", 
        "prev_yards_after_catch", "prev_receiving_air_yards", "prev_targets"
    ]

    X = df[columns_to_use]
    y = df["season_fantasy_points_ppr"]
    
    # Split Data into training and testing seasons (ONLY TRAIN ON PREVIOUS SEASONS FROM THE TESTING)
    df = df.sort_values(by=["player_id", "season"])
    train_years = df["season"] <= 2020
    test_years = df["season"] > 2020

    X_train = X[train_years]
    X_test = X[test_years]

    y_train = y[train_years]
    y_test = y[test_years]

    # Scale features up front so every model receives the same transformed data.
    X_train_scaled, X_test_scaled = transform_data(X_train, X_test)

    # Model has two parts:
    #   1. "model": the actual classifier object from scikit-learn
    #   2. "params": a grid of hyperparameters we want to test during tuning
    # GridSearchCV will automatically try all combinations of these parameters
    model = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [300, 500, 100],
                "max_depth": [None, 10, 20, 30, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
        },
    }

    # Train and tune the  model using 4-Fold Time Series Validation to evaluate accuracy
    # This helps us get a more reliable estimate of performance than a single train/test split.
    for name, cfg in model.items():
        print(f"\n--- {name} ---")

        model = cfg["model"]
        param_grid = cfg["params"]

        # Use TimeSeriesSplit to prevent data leakage from future seasons for cv scores. Time-series data needs specific ordering.
        tscv = TimeSeriesSplit(n_splits=4)

        # Check performance with default parameters and use R^2 as the scoring metric for regression
        baseline_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring="r2")
        baseline_mean = baseline_scores.mean()
        baseline_std = baseline_scores.std()
        print(f"Baseline mean R^2: {baseline_mean:.3f}")

        # GridSearchCV for hyperparameter tuning
        grid = GridSearchCV(model, param_grid, cv=tscv, scoring="r2", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        print("Best parameters:", grid.best_params_)
        print(f"Tuned mean R^2: {grid.best_score_:.3f}")

        y_pred = grid.predict(X_train_scaled)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        print(f"Training RMSE: {rmse:.2f}, MAE: {mae:.2f}")


    random_param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": [10, 20, 30, 40, 50, None],
        "min_samples_split": randint(2, 20),
        "min_samples_leaf": randint(1, 10),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True],
        "max_samples": uniform(0.7, 0.3),
    }

    
    # Create a fresh Random Forest model for RandomizedSearchCV
    # We create a new instance to ensure both search methods start from the same baseline
    model_random = RandomForestRegressor(random_state=42)

    # Perform RandomizedSearchCV
    # - n_iter=50: Try 50 random parameter combinations (much fewer than grid search)
    #   This is a good balance between exploration and computation time.
    random_search = RandomizedSearchCV(
        model_random,
        random_param_dist,
        n_iter=50,  # Number of random parameter combinations to try
        cv=4,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    # Fit the randomized search on training data
    print("\nStarting RandomizedSearchCV...")
    random_search.fit(X_train_scaled, y_train)
    print("RandomizedSearchCV completed!\n")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")
    print(f"Improvement over baseline: {random_search.best_score_ - baseline_mean:.4f}")

    # ============================================================================
    # SAVE THE BEST MODEL
    # ============================================================================
    # Compare the results from both search methods
    print(f"\nBaseline accuracy:           {baseline_mean:.4f}")
    print(f"GridSearchCV best accuracy:  {grid.best_score_:.4f}")
    print(f"RandomizedSearchCV accuracy: {random_search.best_score_:.4f}")

    # Select the best model based on cross-validation score
    if grid.best_score_ >= random_search.best_score_:
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_
        search_method = "GridSearchCV"
    else:
        best_model = random_search.best_estimator_
        best_params = random_search.best_params_
        best_score = random_search.best_score_
        search_method = "RandomizedSearchCV"

    print(f"\nBest model found using: {search_method}")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation accuracy: {best_score:.4f}")

    # Extract feature importances from the trained Random Forest
    # Features that contribute more to reducing prediction error have higher importance values
    importances = pd.Series(best_model.feature_importances_, index=columns_to_use).sort_values(ascending=False)
    print("\nTop 10 Random Forest feature importances:")
    print(importances.head(10))

    # ============================================================================
    # EVALUATE ON TEST SET
    # ============================================================================
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"\nTest set accuracy: {test_accuracy:.4f}")
    print(f"Cross-validation accuracy: {best_score:.4f}")
    print(f"Difference: {abs(test_accuracy - best_score):.4f}")

    # ============================================================================
    # SAVE THE BEST MODEL
    # ============================================================================
    # Create a directory to save models if it doesn't exist
    # We save models in a 'models' directory to keep the project organized.
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Save the best model using joblib (recommended by scikit-learn)
    # joblib is more efficient than pickle for sklearn models with large numpy arrays.
    # The model file can be loaded later using: joblib.load('model_filename.joblib')
    model_filename = os.path.join(model_dir, "best_random_forest_model.joblib")
    joblib.dump(best_model, model_filename)
    print(f"Model saved to: {model_filename}")

if __name__ == "__main__":
    main()
