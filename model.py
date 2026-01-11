# %%
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import shutil
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs detected: {gpus}")
else:
    print("No GPU detected, running on CPU.")

# ----------------------------
# Configuration
# ----------------------------
# Path to your preprocessed data directory
PATH = os.path.join('data')  
# Adjust if needed, e.g.:
# PATH = r"C:\Users\hamdi\Sign-Language-Translator\data_preprocessed"

# ----------------------------
# 1) Gather Sign Labels Dynamically
# ----------------------------
# Each subfolder in PATH is considered a separate sign
actions = sorted([
    d for d in os.listdir(PATH) 
    if os.path.isdir(os.path.join(PATH, d))
])
actions = np.array(actions)

print(f"Found {len(actions)} signs: {actions}")

# Create a label map to map each action label to a numeric value
label_map = {label: idx for idx, label in enumerate(actions)}

# ----------------------------
# 2) Load the Landmarks & Labels
# ----------------------------
landmarks = []
labels = []

# Iterate over each sign
for action in actions:
    action_folder = os.path.join(PATH, action)
    
    # Each subfolder inside action_folder is a 'sequence'
    sequence_folders = sorted([
        sf for sf in os.listdir(action_folder) 
        if os.path.isdir(os.path.join(action_folder, sf))
    ])
    
    for seq_folder in sequence_folders:
        seq_path = os.path.join(action_folder, seq_folder)
        
        # Within each sequence folder, load each frame .npy file
        frame_files = sorted([
            f for f in os.listdir(seq_path) if f.endswith('.npy')
        ])
        
        temp = []
        for frame_file in frame_files:
            frame_path = os.path.join(seq_path, frame_file)
            
            # --- Optional checks for empty/corrupted files ---
            if os.path.getsize(frame_path) == 0:
                print(f"Skipping empty file: {frame_path}")
                continue
            
            try:
                frame_data = np.load(frame_path, allow_pickle=True)
            except (EOFError, ValueError) as e:
                print(f"Skipping invalid .npy file: {frame_path} ({e})")
                continue
            # -------------------------------------------------
            
            temp.append(frame_data)
        
        # OPTIONAL: If you *must* have exactly 10 frames per sequence:
        # if len(temp) != 10:
        #     print(f"Skipping {seq_path} because it doesn't have exactly 10 valid frames.")
        #     continue

        if len(temp) == 0:
            # If no valid frames remain, skip this sequence
            continue

        landmarks.append(temp)
        labels.append(label_map[action])

# Convert to NumPy arrays
X = np.array(landmarks)  # shape: (num_samples, num_frames, num_features)
Y = to_categorical(labels, num_classes=len(actions)).astype(int)

print(f"Data shape: X={X.shape}, Y={Y.shape}")

# ----------------------------
# 3) Train-Test Split
# ----------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.10, random_state=34, stratify=Y
)

# ----------------------------
# 4) Define the Model
# ----------------------------
model = Sequential()
# Adjust input_shape based on how many frames and features you have
# If you're always collecting 10 frames with 126 features each, it's (10, 126).
# If you have a different number of frames or features, update accordingly.
model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(LSTM(32, return_sequences=False, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("\nStarting training...")
model.fit(X_train, Y_train, epochs=100)

# ----------------------------
# 5) Save the Trained Model
# ----------------------------
model.save('my_model.h5')
print("Model saved as my_model.h5")

# ----------------------------
# 6) Evaluate
# ----------------------------
predictions = np.argmax(model.predict(X_test, batch_size=32), axis=1)
test_labels = np.argmax(Y_test, axis=1)
accuracy = metrics.accuracy_score(test_labels, predictions)

print(f"Test Accuracy: {accuracy:.4f}")
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(test_labels, predictions)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(test_labels, predictions))

