# %%
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import mlflow
import mlflow.tensorflow

# MLflow configuration
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow server URL
mlflow.set_experiment("sign-language-recognition")

# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
else:
    print("No GPU detected, running on CPU.")

# ----------------------------
# Configuration
# ----------------------------
PATH = os.path.join('data')

if not os.path.exists(PATH):
    raise FileNotFoundError(f"The path '{PATH}' does not exist. Please run data_collection.py first.")

# ----------------------------
# 1) Gather Sign Labels Dynamically
# ----------------------------
actions = sorted([
    d for d in os.listdir(PATH) 
    if os.path.isdir(os.path.join(PATH, d))
])
actions = np.array(actions)

print(f"Found {len(actions)} signs: {actions}")

if len(actions) == 0:
    raise ValueError(f"No sign folders found in '{PATH}'. Please run data_collection.py first.")

label_map = {label: idx for idx, label in enumerate(actions)}

# ----------------------------
# 2) Load the Landmarks & Labels
# ----------------------------
landmarks = []
labels = []

for action in actions:
    action_folder = os.path.join(PATH, action)
    sequence_folders = sorted([
        sf for sf in os.listdir(action_folder) 
        if os.path.isdir(os.path.join(action_folder, sf))
    ])
    
    print(f"Processing sign '{action}': {len(sequence_folders)} sequences")
    
    for seq_folder in sequence_folders:
        seq_path = os.path.join(action_folder, seq_folder)
        frame_files = sorted([
            f for f in os.listdir(seq_path) if f.endswith('.npy')
        ])
        
        temp = []
        for frame_file in frame_files:
            frame_path = os.path.join(seq_path, frame_file)
            
            if os.path.getsize(frame_path) == 0:
                print(f"Skipping empty file: {frame_path}")
                continue
            
            try:
                frame_data = np.load(frame_path, allow_pickle=True)
            except (EOFError, ValueError) as e:
                print(f"Skipping invalid .npy file: {frame_path} ({e})")
                continue
            
            temp.append(frame_data)
        
        if len(temp) != 10:
            print(f"Warning: {seq_path} has {len(temp)} frames instead of 10")
            if len(temp) == 0:
                continue

        landmarks.append(temp)
        labels.append(label_map[action])

if len(landmarks) == 0:
    raise ValueError("No training data found! Please run data_collection.py to collect sign data first.")

X = np.array(landmarks)
Y = to_categorical(labels, num_classes=len(actions)).astype(int)

print(f"\nData shape: X={X.shape}, Y={Y.shape}")
print(f"Total samples: {len(X)}")

# ----------------------------
# 3) Train-Test Split
# ----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, random_state=34, stratify=Y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ----------------------------
# 4) MLflow Training Run
# ----------------------------
with mlflow.start_run(run_name="LSTM_Sign_Language_Model"):
    
    # Log parameters
    mlflow.log_param("num_signs", len(actions))
    mlflow.log_param("num_samples", len(X))
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("test_size", 0.10)
    mlflow.log_param("random_state", 34)
    mlflow.log_param("epochs", 20)
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("loss", "categorical_crossentropy")
    
    # Model architecture parameters
    mlflow.log_param("lstm_layer_1", 32)
    mlflow.log_param("lstm_layer_2", 64)
    mlflow.log_param("lstm_layer_3", 32)
    mlflow.log_param("dense_layer", 32)
    mlflow.log_param("num_frames", X.shape[1])
    mlflow.log_param("num_features", X.shape[2])
    
    # Log signs
    mlflow.log_param("signs", ",".join(actions))
    
    # Define the Model
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(LSTM(32, return_sequences=False, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    print("\nModel summary:")
    model.summary()
    
    # Log model architecture
    mlflow.log_text(str(model.to_json()), "model_architecture.json")
    
    print("\nStarting training...")
    
    # Train with callback to log metrics
    class MLflowCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            mlflow.log_metric("train_loss", logs['loss'], step=epoch)
            mlflow.log_metric("train_accuracy", logs['categorical_accuracy'], step=epoch)
            if 'val_loss' in logs:
                mlflow.log_metric("val_loss", logs['val_loss'], step=epoch)
                mlflow.log_metric("val_accuracy", logs['val_categorical_accuracy'], step=epoch)
    
    history = model.fit(
        X_train, Y_train, 
        epochs=20, 
        validation_split=0.1, 
        verbose=1,
        callbacks=[MLflowCallback()]
    )
    
    # Save the Model
    model.save('my_model.h5')
    print("\nModel saved as my_model.h5")
    
    # Log model to MLflow
    mlflow.tensorflow.log_model(model, "model")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions = np.argmax(model.predict(X_test, batch_size=32), axis=1)
    test_labels = np.argmax(Y_test, axis=1)
    
    # Calculate metrics
    accuracy = metrics.accuracy_score(test_labels, predictions)
    precision = metrics.precision_score(test_labels, predictions, average='weighted', zero_division=0)
    recall = metrics.recall_score(test_labels, predictions, average='weighted', zero_division=0)
    f1 = metrics.f1_score(test_labels, predictions, average='weighted', zero_division=0)
    
    # Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, classification_report
    
    cm = confusion_matrix(test_labels, predictions)
    print("\nConfusion Matrix:\n", cm)
    
    # Save and log confusion matrix
    np.save('confusion_matrix.npy', cm)
    mlflow.log_artifact('confusion_matrix.npy')
    
    # Classification report
    report = classification_report(test_labels, predictions, target_names=actions)
    print("\nClassification Report:\n", report)
    
    # Save and log classification report
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    mlflow.log_artifact('classification_report.txt')
    
    # Log the saved model file
    mlflow.log_artifact('my_model.h5')
    
    print("\n✅ Training complete! Check MLflow UI for results.")
    print(f"Run ID: {mlflow.active_run().info.run_id}")