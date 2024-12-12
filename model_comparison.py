import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb

def evaluate_model_with_time(model, val_x, val_y, training_time):
    """
    Evaluate model performance including execution time
    """
    start_time = time.time()
    predictions = model.predict(val_x)
    inference_time = time.time() - start_time

    if len(predictions.shape) == 1:
        # Reshape if model outputs flat predictions
        predictions = predictions.reshape(-1, 2)

    dx = predictions[:, 0] - val_y[:, 0]
    dy = predictions[:, 1] - val_y[:, 1]
    distances = np.sqrt(dx**2 + dy**2)

    metrics = {
        "Mean Distance Error": np.mean(distances),
        "Median Distance Error": np.median(distances),
        "90th Percentile Error": np.percentile(distances, 90),
        "Training Time (s)": training_time,
        "Inference Time (s)": inference_time,
        "Inference Time per Sample (ms)": (inference_time / len(val_x)) * 1000,
        "RMSE": np.sqrt(np.mean(distances**2)),
        "MSE": np.mean(distances**2)
    }

    return metrics, distances, predictions

def train_and_evaluate_traditional_models(X_train, y_train, X_val, y_val):
    """
    Train and evaluate traditional ML models using pre-split data
    """
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
        "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10)
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()

        if name in ["XGBoost", "SVR"]:
            # Train separate models for x and y coordinates
            if name == "XGBoost":
                model_x = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                model_y = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            else:  # SVR
                model_x = SVR(kernel='rbf', C=1.0, epsilon=0.1)
                model_y = SVR(kernel='rbf', C=1.0, epsilon=0.1)

            model_x.fit(X_train, y_train[:, 0])
            model_y.fit(X_train, y_train[:, 1])

            training_time = time.time() - start_time

            # Predict
            start_inference = time.time()
            pred_x = model_x.predict(X_val)
            pred_y = model_y.predict(X_val)
            predictions = np.column_stack((pred_x, pred_y))
            inference_time = time.time() - start_inference

        else:
            # Train a single model for both coordinates
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            start_inference = time.time()
            predictions = model.predict(X_val)
            inference_time = time.time() - start_inference

        dx = predictions[:, 0] - y_val[:, 0]
        dy = predictions[:, 1] - y_val[:, 1]
        distances = np.sqrt(dx**2 + dy**2)

        metrics = {
            "Mean Distance Error": np.mean(distances),
            "Median Distance Error": np.median(distances),
            "90th Percentile Error": np.percentile(distances, 90),
            "Training Time (s)": training_time,
            "Inference Time (s)": inference_time,
            "Inference Time per Sample (ms)": (inference_time / len(X_val)) * 1000,
            "RMSE": np.sqrt(np.mean(distances**2)),
            "MSE": np.mean(distances**2),
            "predictions": predictions
        }

        results[name] = metrics

        print(f"\n{name} Performance:")
        for metric, value in metrics.items():
            if metric != "predictions":
                print(f"{metric}: {value:.4f}")

    return results

# Function to format results as a DataFrame
def format_results(results):
    """Convert results dictionary to a pandas DataFrame for easy comparison"""
    return pd.DataFrame(results).round(4)
