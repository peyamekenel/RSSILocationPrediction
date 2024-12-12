import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from model_comparison import train_and_evaluate_traditional_models, format_results
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def create_encoder(input_shape):
    inputs = Input(shape=input_shape, name="encoder_input")
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    encoded = Conv1D(16, kernel_size=3, activation='relu', padding='same', name="encoded_output")(x)
    encoder = Model(inputs, encoded, name="encoder")
    return encoder

def create_autoencoder(encoder):
    encoded_input = encoder.input
    encoded_output = encoder.output
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(encoded_output)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    outputs = Conv1D(1, kernel_size=3, activation='linear', padding='same')(x)
    autoencoder = Model(encoded_input, outputs, name="autoencoder")
    autoencoder.compile(optimizer=Adam(0.001), loss='mse')
    return autoencoder

def create_location_predictor(encoder, trainable_encoder=False):
    for layer in encoder.layers:
        layer.trainable = trainable_encoder

    encoded_output = encoder.output
    x = Flatten()(encoded_output)
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    predictions = Dense(2, activation='linear')(x)

    rmse_metric = tf.keras.metrics.RootMeanSquaredError(name='rmse')
    model = Model(encoder.input, predictions, name="location_predictor")
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=[rmse_metric, 'mse'])
    return model

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    return wrapper

@measure_time
def train_neural_network(train_x, train_y, val_x, val_y, unlabeled_x, input_shape):
    encoder = create_encoder(input_shape)
    autoencoder = create_autoencoder(encoder)

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    mc = ModelCheckpoint('best_autoencoder.keras', monitor='val_loss', save_best_only=True, verbose=1)

    print("Training autoencoder on unlabeled data...")
    autoencoder.fit(
        unlabeled_x, unlabeled_x,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es, rlr, mc],
        verbose=1
    )

    autoencoder.load_weights('best_autoencoder.keras')

    print("\nTraining location predictor...")
    predictor = create_location_predictor(encoder, trainable_encoder=False)
    predictor.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=200,
        batch_size=32,
        callbacks=[es, rlr],
        verbose=1
    )

    print("\nFine-tuning complete model...")
    predictor = create_location_predictor(encoder, trainable_encoder=True)
    predictor.fit(
        train_x, train_y,
        validation_data=(val_x, val_y),
        epochs=200,
        batch_size=32,
        callbacks=[es, rlr],
        verbose=1
    )

    return predictor

def plot_metric_comparison(results, metric_name):
    plt.figure(figsize=(10, 6))

    # Extract values and model names
    models = list(results.keys())
    values = [results[model][metric_name] for model in models]

    # Define metrics where lower values are better
    lower_is_better = any(term in metric_name.lower() for term in ["error", "time", "rmse", "mse"])

    # Find the best value and its index
    if lower_is_better:
        best_value = min(values)
        best_idx = values.index(best_value)
    else:
        best_value = max(values)
        best_idx = values.index(best_value)

    # Create color list with best performer highlighted
    colors = ['lightgray'] * len(values)
    colors[best_idx] = 'green'

    # Create bar plot
    bars = plt.bar(models, values, color=colors)
    plt.title(f'{metric_name} Comparison')
    plt.xlabel('Model')
    plt.ylabel(metric_name)
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    # Highlight best performer in legend
    best_model = models[best_idx]
    plt.text(0.98, 0.98, f'Best: {best_model}\nValue: {best_value:.4f}',
             transform=plt.gca().transAxes, ha='right', va='top',
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f'{metric_name.lower().replace(" ", "_")}.png')
    plt.close()

def plot_location_predictions(y_true, predictions, model_name):
    plt.scatter(y_true[:, 0], y_true[:, 1], c='blue', label='Gerçek Konumlar', alpha=0.6)
    plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Tahmin Edilen Konumlar', alpha=0.6)
    plt.title(f'{model_name}')
    plt.xlabel('Normalize X Koordinatı')
    plt.ylabel('Normalize Y Koordinatı')
    plt.legend()
    plt.grid(True)

def plot_all_predictions(val_data, predictions_dict):
    """Create a single figure with all model predictions side by side"""
    plt.figure(figsize=(15, 10))
    plt.suptitle('Model Konum Tahminleri Karşılaştırması', fontsize=14, y=1.02)

    model_names = {
        "Neural Network": "Sinir Ağı",
        "Random Forest": "Rastgele Orman",
        "KNN": "KNN",
        "XGBoost": "XGBoost",
        "SVR": "SVR",
        "Decision Tree": "Karar Ağacı"
    }

    for idx, (model_name, predictions) in enumerate(predictions_dict.items()):
        plt.subplot(2, 3, idx + 1)
        y_true = val_data["nn"] if model_name == "Neural Network" else val_data["traditional"]

        plt.scatter(y_true[:, 0], y_true[:, 1], c='blue', label='Gerçek', alpha=0.6, s=20)
        plt.scatter(predictions[:, 0], predictions[:, 1], c='red', label='Tahmin', alpha=0.6, s=20)
        plt.title(model_names[model_name])
        plt.xlabel('Normalize X')
        plt.ylabel('Normalize Y')
        plt.legend(fontsize=8)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('location_predictions_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

# PLACEHOLDER: Next function (main)

def main():
    SEED = 42
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Loading and preprocessing data...")
    labeled_df = pd.read_csv('~/attachments/iBeacon_RSSI_Labeled.csv')
    unlabeled_df = pd.read_csv('~/attachments/iBeacon_RSSI_Unlabeled.csv')

    beacon_columns = [col for col in labeled_df.columns if col.startswith('b300')]

    default_fill_value = -150
    for col in beacon_columns:
        labeled_df[col] = labeled_df[col].replace(-200, np.nan).fillna(default_fill_value)
        unlabeled_df[col] = unlabeled_df[col].replace(-200, np.nan).fillna(default_fill_value)

    combined_beacons = pd.concat([labeled_df[beacon_columns], unlabeled_df[beacon_columns]])
    rssi_scaler = MinMaxScaler()
    rssi_scaler.fit(combined_beacons)
    labeled_df[beacon_columns] = rssi_scaler.transform(labeled_df[beacon_columns])
    unlabeled_df[beacon_columns] = rssi_scaler.transform(unlabeled_df[beacon_columns])

    labeled_df['x'] = labeled_df['location'].str[0]
    labeled_df['y'] = labeled_df['location'].str[1:].astype(int)
    x_encoder = LabelEncoder()
    labeled_df['x_encoded'] = x_encoder.fit_transform(labeled_df['x'])

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    labeled_df['x_normalized'] = x_scaler.fit_transform(labeled_df[['x_encoded']])
    labeled_df['y_normalized'] = y_scaler.fit_transform(labeled_df[['y']])

    X = labeled_df[beacon_columns].values
    y = labeled_df[['x_normalized', 'y_normalized']].values
    X_reshaped = X.reshape(-1, len(beacon_columns), 1)
    unlabeled_x = unlabeled_df[beacon_columns].values.reshape(-1, len(beacon_columns), 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(X_reshaped, y, test_size=0.2, random_state=SEED)

    print("\nTraining Neural Network...")
    input_shape = (len(beacon_columns), 1)
    nn_model, nn_training_time = train_neural_network(
        X_train_nn, y_train_nn, X_val_nn, y_val_nn, unlabeled_x, input_shape
    )

    start_time = time.time()
    nn_predictions = nn_model.predict(X_val_nn)
    nn_inference_time = time.time() - start_time

    dx = nn_predictions[:, 0] - y_val_nn[:, 0]
    dy = nn_predictions[:, 1] - y_val_nn[:, 1]
    nn_distances = np.sqrt(dx**2 + dy**2)

    nn_metrics = {
        "Mean Distance Error": np.mean(nn_distances),
        "Median Distance Error": np.median(nn_distances),
        "90th Percentile Error": np.percentile(nn_distances, 90),
        "Training Time (s)": nn_training_time,
        "Inference Time (s)": nn_inference_time,
        "Inference Time per Sample (ms)": (nn_inference_time / len(X_val_nn)) * 1000,
        "RMSE": np.sqrt(np.mean(nn_distances**2)),
        "MSE": np.mean(nn_distances**2)
    }

    traditional_results = train_and_evaluate_traditional_models(X_train, y_train, X_val, y_val)

    all_results = {"Neural Network": nn_metrics, **traditional_results}

    comparison_df = format_results(all_results)
    print("\nModel Comparison Results:")
    print(comparison_df)

    comparison_df.to_csv('model_comparison_results.csv')

    # Generate individual metric comparisons
    for metric in ['Mean Distance Error', 'Median Distance Error', '90th Percentile Error', 'Training Time (s)', 'Inference Time (s)', 'RMSE', 'MSE']:
        plot_metric_comparison(all_results, metric)

    # Generate location prediction plots
    predictions_dict = {
        "Neural Network": nn_predictions,
        "Random Forest": np.array(traditional_results["Random Forest"]["predictions"]),
        "KNN": np.array(traditional_results["KNN"]["predictions"]),
        "XGBoost": np.array(traditional_results["XGBoost"]["predictions"]),
        "SVR": np.array(traditional_results["SVR"]["predictions"]),
        "Decision Tree": np.array(traditional_results["Decision Tree"]["predictions"])
    }
    val_data = {
        "nn": y_val_nn,
        "traditional": y_val
    }
    plot_all_predictions(val_data, predictions_dict)

    plt.figure(figsize=(12, 6))
    for model_name, metrics in all_results.items():
        if model_name == "Neural Network":
            distances = nn_distances
        else:
            dx = traditional_results[model_name]["predictions"][:, 0] - y_val[:, 0]
            dy = traditional_results[model_name]["predictions"][:, 1] - y_val[:, 1]
            distances = np.sqrt(dx**2 + dy**2)
        sns.kdeplot(distances, label=model_name)

    plt.title('Error Distribution Comparison')
    plt.xlabel('Distance Error')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('error_distributions.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    training_times = [metrics["Training Time (s)"] for metrics in all_results.values()]
    plt.bar(all_results.keys(), training_times)
    plt.title('Training Time Comparison')
    plt.xlabel('Model')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('training_times.png')
    plt.close()

if __name__ == "__main__":
    main()
