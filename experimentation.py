import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ydata_profiling import ProfileReport
from xgboost import XGBRegressor

def rb_feature_engineering(df):
    # Sort by player and season
    df = df.sort_values(by=["player_id", "season"])

    # Add prev season stats
    df["prev_season_fantasy_points_ppr"] = df.groupby("player_id")["season_fantasy_points_ppr"].shift(1)
    df["prev_season_rushing_yards"] = df.groupby("player_id")["season_rushing_yards"].shift(1)
    df["prev_season_rush_touchdown"] = df.groupby("player_id")["season_rush_touchdown"].shift(1)
    df["prev_season_rush_attempts"] = df.groupby("player_id")["season_rush_attempts"].shift(1)
    df["prev_games_played_season"] = df.groupby("player_id")["games_played_season"].shift(1)

     # Change year over year
    df['delta_fantasy_points'] = df.groupby('player_id')['season_fantasy_points_ppr'].diff().clip(-100, 100)

    # Add column for predicted stats
    df["fantasy_points_ppr_next"] = df.groupby("player_id")["season_fantasy_points_ppr"].shift(-1)

    # Drop rows where previous season stats are missing (first season for each player)
    df = df.dropna(subset=[
        "prev_season_fantasy_points_ppr", 
        "prev_season_rushing_yards", 
        "prev_season_rush_touchdown", 
        "prev_season_rush_attempts",
        "prev_games_played_season",
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
    # Read data
    main_df = pd.read_csv("yearly_player_stats_offense.csv")
    main_df = main_df[main_df["season_type"] == "REG"]

    rb_df = main_df[main_df["position"] == "RB"]
    # wr_df = main_df[main_df["position"] == "WR"]
    # qb_df = main_df[main_df["position"] == "QB"]
    # te_df = main_df[main_df["position"] == "TE"]

    rb_df = rb_feature_engineering(rb_df)

    rb_columns_to_use = [
        "age", "draft_ovr", "years_exp", "prev_season_fantasy_points_ppr", 
        "prev_season_rushing_yards", "prev_season_rush_touchdown", "prev_season_rush_attempts", 
        "prev_games_played_season"
    ]

    X = rb_df[rb_columns_to_use]
    y = rb_df["season_fantasy_points_ppr"]
    
    # Split Data into training and testing seasons (ONLY TRAIN ON PREVIOUS SEASONS FROM THE TESTING)
    rb_df = rb_df.sort_values(by=["player_id", "season"])
    train_years = rb_df["season"] <= 2020
    test_years = rb_df["season"] > 2020

    X_train = X[train_years]
    X_test = X[test_years]

    y_train = y[train_years]
    y_test = y[test_years]


    # This uses a combination of positions and the actual statistics to predict themselves.
    columns_to_use = [
        "position", "passing_yards", "pass_touchdown", "interception",
        "rushing_yards", "rush_touchdown", "receptions", "receiving_yards", "receiving_touchdown",
        "targets", "touches", "age", "draft_year"
    ]
    X = main_df[columns_to_use]
    y = main_df["fantasy_points_ppr"]

    # Convert the position columns into binary features; NOTE: Model has been split by position
    # X = pd.get_dummies(X, columns=["position"], drop_first=True)
    

    # Split data into train and test subsets (80% train - 20% test)
    # Although we won't use the test set in this file, it is important
    # to hold out a set of test data the model does not see when training
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale features up front so every model receives the same transformed data.
    X_train_scaled, X_test_scaled = transform_data(X_train, X_test)

    # Each model has two parts:
    #   1. "model": the actual classifier object from scikit-learn
    #   2. "params": a grid of hyperparameters we want to test during tuning
    # GridSearchCV will automatically try all combinations of these parameters
    models = {
        "Ridge Regression": {
            "model": Ridge(),
            "params": {
                "alpha": [0.1, 1.0, 10.0],
                "solver": ["auto", "svd", "cholesky"]
            },
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            },
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42, objective="reg:squarederror"),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [3, 6, 10],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0]
            },
        },
    }

    # Train and tune each model using 4-Fold Time Series Validation
    # For each model, we'll start by training it with the default parameters
    # to see how it performs without tuning.
    # Then, we use 4-Fold Time Series Validation to evaluate its accuracy.
    # This helps us get a more reliable estimate of performance than a single train/test split.
    # Finally, we average the accuracy across all four folds to get a single estimate of performance.
    for name, cfg in models.items():
        print(f"\n--- {name} ---")

        model = cfg["model"]
        param_grid = cfg["params"]

        # Use TimeSeriesSplit to prevent data leakage from future seasons for cv scores. Time-series data needs specific ordering.
        tscv = TimeSeriesSplit(n_splits=4)

        # Check performance with default parameters
        # Use R^2 as the scoring metric for regression
        base_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring="r2")
        print(f"Baseline mean R^2: {base_scores.mean():.3f}")

        # GridSearchCV for hyperparameter tuning
        grid = GridSearchCV(model, param_grid, cv=tscv, scoring="r2", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        print("Best parameters:", grid.best_params_)
        print(f"Tuned mean R^2: {grid.best_score_:.3f}")

        y_pred = grid.predict(X_train_scaled)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        print(f"Training RMSE: {rmse:.2f}, MAE: {mae:.2f}")


    # Extract feature importances from the trained Random Forest
    # Features that contribute more to reducing prediction error have higher importance values
    rf_model = grid.best_estimator_ if name == "Random Forest" else None
    if rf_model:
        importances = pd.Series(rf_model.feature_importances_, index=rb_columns_to_use).sort_values(ascending=False)
        print("Top 10 Random Forest feature importances:")
        print(importances.head(10))

    # Now lets try to predict 2025 based on training from before 2024
    train_df = rb_df[rb_df["season"] < 2024].copy()
    train_df = train_df.dropna(subset=["fantasy_points_ppr_next"])
    test_df = rb_df[rb_df["season"] == 2024].copy()

    X_train = train_df[rb_columns_to_use]
    y_train = train_df["fantasy_points_ppr_next"]

    final_model = GridSearchCV(RandomForestRegressor(random_state=42), models["Random Forest"]["params"], cv=tscv, scoring="r2", n_jobs=-1)
    final_model.fit(X_train, y_train)

    X_test = test_df[rb_columns_to_use]
    pred_2025 = final_model.predict(X_test)
    test_df.loc[:, "predicted_2025_points"] = pred_2025

    # Show results
    print("\n2025 Season Predictions for RBs: \n")
    print(
        test_df[["player_name", "season", "predicted_2025_points"]]
        .sort_values(by="predicted_2025_points", ascending=False)
        .head(20)
    )

if __name__ == "__main__":
    main()
