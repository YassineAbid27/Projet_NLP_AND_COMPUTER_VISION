import os
import numpy as np
import pandas as pd
from glob import glob

# ----------------------------
# Configuration
# ----------------------------
TRAIN_CSV = r"C:\Users\hamdi\Desktop\ASL\asl-signs\train.csv"
TRAIN_LANDMARK_FILES = r"C:\Users\hamdi\Desktop\ASL\asl-signs\train_landmark_files"
OUTPUT_DIR = r"C:\Users\hamdi\Sign-Language-Translator\data_preprocessed"

# Number of frames to sample per Parquet file
FRAMES_PER_SEQUENCE = 10

# We'll only pivot on these 'type' values
TYPES_OF_INTEREST = ["left_hand", "right_hand"]

# ----------------------------
# Step 1: Build seq_id -> sign mapping from train.csv
# ----------------------------
print("Loading training CSV...")
train_df = pd.read_csv(TRAIN_CSV)

# Confirm that 'sequence_id' and 'sign' columns exist
expected_cols = {"sequence_id", "sign"}
actual_cols = set(train_df.columns)
missing_cols = expected_cols - actual_cols
if missing_cols:
    raise ValueError(f"CSV is missing columns: {missing_cols}. Found columns: {actual_cols}")

# Convert sequence_id to string so it matches filenames like "1000035562"
train_df["sequence_id"] = train_df["sequence_id"].astype(str)

# Build a dictionary { "1000035562": "blow", "1000106739": "wait", ... }
seq_to_sign = dict(zip(train_df["sequence_id"], train_df["sign"]))

print(f"Found {len(seq_to_sign)} sequence-to-sign mappings.")

# ----------------------------
# Step 2: Create output directory
# ----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Step 3: Process each participant folder, but determine sign from each Parquet file
# ----------------------------
all_participant_folders = [
    f for f in os.listdir(TRAIN_LANDMARK_FILES)
    if os.path.isdir(os.path.join(TRAIN_LANDMARK_FILES, f))
]
print(f"Found {len(all_participant_folders)} participant folders in '{TRAIN_LANDMARK_FILES}'.")

sequence_count = 0

for participant_folder in all_participant_folders:
    participant_path = os.path.join(TRAIN_LANDMARK_FILES, participant_folder)
    
    # Get all .parquet files for this participant
    parquet_files = sorted(glob(os.path.join(participant_path, "*.parquet")))
    if not parquet_files:
        continue
    
    print(f"\nParticipant folder '{participant_folder}' has {len(parquet_files)} Parquet files.")
    
    for parquet_file in parquet_files:
        # Extract the sequence_id from the filename (e.g. "1000035562" from "1000035562.parquet")
        base_name = os.path.splitext(os.path.basename(parquet_file))[0]
        seq_id_str = base_name  # e.g. "1000035562"

        # Look up the sign label from seq_id_str
        sign_label = seq_to_sign.get(seq_id_str)
        if sign_label is None:
            print(f"  [Skipping] {parquet_file} => No sign found for sequence_id={seq_id_str}")
            continue
        
        # Create the sign folder in the output directory
        output_sign_folder = os.path.join(OUTPUT_DIR, sign_label)
        os.makedirs(output_sign_folder, exist_ok=True)
        
        # Read the Parquet file
        df = pd.read_parquet(parquet_file)
        
        # Filter for left_hand, right_hand
        df = df[df["type"].isin(TYPES_OF_INTEREST)]
        
        # Pivot so each 'frame' is a row, columns are (type, landmark_index, x/y/z)
        pivoted = df.pivot(
            index="frame",
            columns=["type", "landmark_index"],
            values=["x", "y", "z"]
        )
        
        # Fill missing with 0
        pivoted = pivoted.fillna(0)
        
        # Flatten the MultiIndex columns into single-level
        pivoted.columns = [f"{col[0]}_{col[1]}_{col[2]}" for col in pivoted.columns]
        
        frames_data = pivoted.to_numpy(dtype=np.float32)
        num_frames = frames_data.shape[0]
        
        if num_frames < FRAMES_PER_SEQUENCE:
            print(f"  [Skipping] {parquet_file}: only {num_frames} frames (need {FRAMES_PER_SEQUENCE}).")
            continue
        
        # Uniformly sample frames
        indices = np.linspace(0, num_frames - 1, FRAMES_PER_SEQUENCE, dtype=int)
        sequence_frames = frames_data[indices]  # shape (10, ?)
        
        # Expect 126 columns for both hands (21 landmarks each * 3 coords * 2 hands)
        if sequence_frames.shape[1] != 126:
            print(f"  [Warning] {parquet_file} has {sequence_frames.shape[1]} columns, expected 126.")
        
        # Create a subfolder for this particular sequence (optional)
        sequence_folder = os.path.join(output_sign_folder, seq_id_str)
        os.makedirs(sequence_folder, exist_ok=True)
        
        # Save each of the sampled frames as .npy
        for frame_idx, frame_vector in enumerate(sequence_frames):
            frame_filepath = os.path.join(sequence_folder, f"{frame_idx}.npy")
            np.save(frame_filepath, frame_vector)
        
        sequence_count += 1
        print(f"  [Saved] {parquet_file} => sign '{sign_label}', sequence_id={seq_id_str}")

print(f"\nAll done. Collected {sequence_count} total sequences.")
print(f"Data saved to: {os.path.abspath(OUTPUT_DIR)}")
