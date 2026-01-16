"""
ZenML Pipeline for Sign Language Recognition
Complete ML workflow: data loading, preprocessing, training, evaluation, deployment
"""

import os
import numpy as np
from typing import Tuple, Dict, Any
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
import mlflow

# ----------------------------
# ZenML Steps
# ----------------------------

@step
def load_data(data_path: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load sign language data from files"""
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path '{data_path}' does not exist")
    
    # Get action labels
    actions = sorted([
        d for d in os.listdir(data_path) 
        if os.path.isdir(os.path.join(data_path, d))
    ])
    actions = np.array(actions)
    
    print(f"Found {len(actions)} signs: {actions}")
    
    if len(actions) == 0:
        raise ValueError(f"No sign folders found in '{data_path}'")
    
    label_map = {label: idx for idx, label in enumerate(actions)}
    
    # Load landmarks and labels
    landmarks = []
    labels = []
    
    for action in actions:
        action_folder = os.path.join(data_path, action)
        sequence_folders = sorted([
            sf for sf in os.listdir(action_folder) 
            if os.path.isdir(os.path.join(action_folder, sf))
        ])
        
        for seq_folder in sequence_folders:
            seq_path = os.path.join(action_folder, seq_folder)
            frame_files = sorted([
                f for f in os.listdir(seq_path) if f.endswith('.npy')
            ])
            
            temp = []
            for frame_file in frame_files:
                frame_path = os.path.join(seq_path, frame_file)
                
                if os.path.getsize(frame_path) == 0:
                    continue
                
                try:
                    frame_data = np.load(frame_path, allow_pickle=True)
                    temp.append(frame_data)
                except (EOFError, ValueError):
                    continue
            
            if len(temp) > 0:
                landmarks.append(temp)
                labels.append(label_map[action])
    
    X = np.array(landmarks)
    Y = to_categorical(labels, num_classes=len(actions)).astype(int)
    
    print(f"Loaded data shape: X={X.shape}, Y={Y.shape}")
    
    return X, Y, actions


@step
def split_data(
    X: np.ndarray, 
    Y: np.ndarray, 
    test_size: float = 0.1,
    random_state: int = 34
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets"""
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    return X_train, X_test, Y_train, Y_test


@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    actions: np.ndarray,
    epochs: int = 100,
    lstm_units_1: int = 32,
    lstm_units_2: int = 64,
    lstm_units_3: int = 32,
    dense_units: int = 32
) -> tf.keras.Model:
    """Train LSTM model with MLflow tracking"""
    
    mlflow.tensorflow.autolog()
    
    # Log parameters
    mlflow.log_param("num_signs", len(actions))
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("lstm_layer_1", lstm_units_1)
    mlflow.log_param("lstm_layer_2", lstm_units_2)
    mlflow.log_param("lstm_layer_3", lstm_units_3)
    mlflow.log_param("dense_layer", dense_units)
    
    # Build model
    model = Sequential([
        LSTM(lstm_units_1, return_sequences=True, activation='relu', 
             input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(lstm_units_2, return_sequences=True, activation='relu'),
        LSTM(lstm_units_3, return_sequences=False, activation='relu'),
        Dense(dense_units, activation='relu'),
        Dense(len(actions), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    print("\nTraining model...")
    model.fit(
        X_train, Y_train,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    
    return model


@step
def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    actions: np.ndarray
) -> Dict[str, float]:
    """Evaluate model and return metrics"""
    
    predictions = np.argmax(model.predict(X_test, batch_size=32), axis=1)
    test_labels = np.argmax(Y_test, axis=1)
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(test_labels, predictions)
    precision = metrics.precision_score(test_labels, predictions, average='weighted', zero_division=0)
    recall = metrics.recall_score(test_labels, predictions, average='weighted', zero_division=0)
    f1 = metrics.f1_score(test_labels, predictions, average='weighted', zero_division=0)
    
    # Log to MLflow
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    
    print("\n=== Evaluation Results ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results


@step
def save_model(
    model: tf.keras.Model,
    model_path: str = "my_model.h5"
) -> str:
    """Save trained model to disk"""
    
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Log model artifact to MLflow
    mlflow.log_artifact(model_path)
    
    return model_path


# ----------------------------
# ZenML Pipeline
# ----------------------------

@pipeline(enable_cache=False)
def sign_language_training_pipeline(
    data_path: str = "data",
    test_size: float = 0.1,
    epochs: int = 100
):
    """
    Complete ML pipeline for sign language recognition
    
    Steps:
    1. Load data
    2. Split into train/test
    3. Train model
    4. Evaluate model
    5. Save model
    """
    
    # Load data
    X, Y, actions = load_data(data_path=data_path)
    
    # Split data
    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=test_size)
    
    # Train model
    model = train_model(X_train, Y_train, actions, epochs=epochs)
    
    # Evaluate model
    metrics_dict = evaluate_model(model, X_test, Y_test, actions)
    
    # Save model
    model_path = save_model(model)
    
    return model_path, metrics_dict


# ----------------------------
# Run Pipeline
# ----------------------------

if __name__ == "__main__":
    # Initialize ZenML
    from zenml.client import Client
    
    print("ZenML Client initialized")
    print(f"Active stack: {Client().active_stack_model.name}")
    
    # Run the pipeline
    pipeline_run = sign_language_training_pipeline(
        data_path="data",
        test_size=0.1,
        epochs=100
    )
    
    print("\n✅ Pipeline completed successfully!")
    print(f"Check ZenML dashboard: zenml up")
    print(f"Check MLflow UI: mlflow ui")