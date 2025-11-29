import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
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
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def main():
    # Read data
    main_df = pd.read_csv("yearly_player_stats_offense.csv")
    main_df = main_df[main_df["season_type"] == "REG"]

    """"
    # Display df general information
    print("\nDataFrame info:")
    print(main_df.info())

    # Create profile report to visualize and explore the dataset (IF NEEDED. Kaggle has most of the dataset information, just hard to work with)
    df = pd.concat([X, y], axis=1) # combine features and target into one dataframe
    profile = ProfileReport(df, title="Player Stats Report")
    profile.to_file("data_report.html")
    """

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

    # Scale features up front so every model receives the same transformed data.
    # This is important for models that are sensitive to feature scaling, such as neural networks.
    # Whenever you transform your data, you must transform your test data the same way.
    # We use a function to scale the data so we can reuse it in the future. This will be critical
    # when we deploy the model and need to apply the same transformation to new data.
    X_train_scaled, X_test_scaled = transform_data(X_train, X_test)

    y_train = y[train_years]
    y_test = y[test_years]


    """ This uses a combination of positions and the actual statistics to predict themselves.
    columns_to_use = [
        "position", "passing_yards", "pass_touchdown", "interception",
        "rushing_yards", "rush_touchdown", "receptions", "receiving_yards", "receiving_touchdown",
        "targets", "touches", "age", "draft_year"
    ]
    X = main_df[columns_to_use]
    y = main_df["fantasy_points_ppr"]
    """

    # Convert the position columns into binary features; NOTE: Model has been split by position
    # X = pd.get_dummies(X, columns=["position"], drop_first=True)
    

    # Split data into train and test subsets (80% train - 20% test)
    # Although we won't use the test set in this file, it is important
    # to hold out a set of test data the model does not see when training
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree Classifier model
    # model = DecisionTreeRegressor(random_state=42)

    # Define a dictionary of models to compare
    # Each model has two parts:
    #   1. "model": the actual classifier object from scikit-learn
    #   2. "params": a grid of hyperparameters we want to test during tuning
    # GridSearchCV will automatically try all combinations of these parameters
    # to find the set that gives the best accuracy.
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

    # Train and tune each model using 4-Fold Cross Validation
    # For each model, we'll start by training it with the default parameters
    # to see how it performs without tuning.
    # Then, we use 4-Fold Cross Validation to evaluate its accuracy.
    # Recall, in 4-Fold Cross Validation, the training data is split into four parts,
    # and the model is trained and validated on different parts each time.
    # This helps us get a more reliable estimate of performance than a single train/test split.
    # Finally, we average the accuracy across all four folds to get a single estimate of performance.
    
    for name, cfg in models.items():
        print(f"\n--- {name} ---")

        model = cfg["model"]
        param_grid = cfg["params"]

        # Check performance with default parameters
        # Use R^2 as the scoring metric for regression
        base_scores = cross_val_score(model, X_train_scaled, y_train, cv=4, scoring="r2")
        print(f"Baseline mean R^2: {base_scores.mean():.3f}")

        # GridSearchCV for hyperparameter tuning
        grid = GridSearchCV(model, param_grid, cv=4, scoring="r2", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)

        print("Best parameters:", grid.best_params_)
        print(f"Tuned mean R^2: {grid.best_score_:.3f}")

    # Perform 4-Fold Cross-Validation to estimate model performance
    # In each iteration, the model is trained on 3/4 of the training data and
    # validated on the remaining 1/4 (validation set). This process repeats 4 times
    # kf = KFold(n_splits=4)
    
    # Use TimeSeriesSplit to prevent data leakage from future seasons for cv scores. Time-series data needs specific ordering.
    # RB fantasy scoring is highly volatile year to year. This can cause negative cross validation scores.
    """"
    tscv = TimeSeriesSplit(n_splits=4)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring="r2")

    # Print the R^2 scores from each fold of the 4-fold cross-validation
    # This shows how well the Decision Tree Regressor performed on different subsets of the training data
    print("Cross-Validation Scores:", cv_scores)

    # Print the mean R^2 score across all folds
    # This gives a single summary metric of model performance during cross-validation
    print(f"Mean Validation Score: {np.mean(cv_scores)}")

    # Initialize a Random Forest Regressor
    # Random Forest trains many decision trees on random subsets of the data and features
    # This helps reduce overfitting and gives a more robust model
    rf = RandomForestRegressor(
        n_estimators=100,     # number of trees in the forest
        max_depth=None,       # trees are grown fully until leaves are pure or contain minimal samples
        random_state=42       # ensures reproducible results
    )

    # Train the Random Forest on the training data
    # The model learns patterns between features (X_train) and target (y_train)
    rf.fit(X_train, y_train)

    # Make predictions on the held-out test set
    # These are the model's estimated fantasy points for players in the test data
    y_pred = rf.predict(X_test)

    # Evaluate the model using R^2 score on the test set
    # R^2 measures how much of the variance in y_test the model can explain (1 = perfect, 0 = no explanation)
    print("R^2 score:", r2_score(y_test, y_pred))
    """

    # Extract feature importances from the trained Random Forest
    # Features that contribute more to reducing prediction error have higher importance values
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Print the top 10 most important features according to the Random Forest
    # This helps identify which stats have the biggest impact on predicting fantasy points
    print("Features Importances:", importances.head(10))

    # Now lets try to predict 2025 based on training from before 2024
    train_df = rb_df[rb_df["season"] < 2024].copy()
    train_df = train_df.dropna(subset=["fantasy_points_ppr_next"])
    test_df = rb_df[rb_df["season"] == 2024].copy()

    X_train = train_df[rb_columns_to_use]
    y_train = train_df["fantasy_points_ppr_next"]
    model.fit(X_train, y_train)

    X_test = test_df[rb_columns_to_use]
    pred_2025 = model.predict(X_test)
    test_df.loc[:, "predicted_2025_points"] = pred_2025

    # Show results
    print(
        test_df[["player_name", "season", "predicted_2025_points"]]
        .sort_values(by="predicted_2025_points", ascending=False)
        .head(20)
    )
    


if __name__ == "__main__":
    main()
