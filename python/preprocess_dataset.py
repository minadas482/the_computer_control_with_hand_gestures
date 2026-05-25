
import pandas as pd
import numpy as np
import os

# 1. Load the Kaggle Dataset
csv_filename = 'gesture_landmarks.csv'
print(f"Loading '{csv_filename}'...")

if not os.path.exists(csv_filename):
    print(f"ERROR: '{csv_filename}' not found!")
    exit()

df = pd.read_csv(csv_filename)

# --- SMART COLUMN DETECTION ---
# We need to find which column holds the text labels (e.g., 'open', 'point')
label_col_name = None

# Check for a column named 'label' or 'gesture'
possible_names = ['label', 'gesture', 'class', 'target']
for name in possible_names:
    if name in df.columns:
        label_col_name = name
        break

# If not found by name, look for the column with String/Text data
if label_col_name is None:
    print("Searching for label column by content...")
    for col in df.columns:
        # Check if the column contains strings (object type)
        if df[col].dtype == 'object':
            label_col_name = col
            break

if label_col_name is None:
    print("CRITICAL ERROR: Could not find a label column (text) in the CSV.")
    print("Columns found:", df.columns.tolist())
    exit()

print(f"FOUND LABEL COLUMN: '{label_col_name}'")

# 2. Define our Mapping
LABEL_MAP = {
    'open': 0,   # IDLE
    'point': 1,  # MOUSE_MOVE
    'peace': 2,  # CLICK
    'thumb': 3,  # SPACEBAR
    'close': 4,  # BACKSPACE (Fist)
    'fist': 4,   # Handle alternate name
    'rock': 4    # Handle alternate name
}

# 3. Filter and Rename
print("Processing data...")
processed_data = []
match_count = 0

for index, row in df.iterrows():
    raw_label = str(row[label_col_name]).lower().strip()
    
    if raw_label in LABEL_MAP:
        new_label = LABEL_MAP[raw_label]
        
        # Get all columns EXCEPT the label column
        # This assumes all other columns are the 63 landmarks
        features = row.drop(label_col_name).values
        
        # Ensure we have exactly 63 numeric features
        if len(features) == 63:
            # Create a new row: [label, lm1, lm2, ... lm63]
            new_row = [new_label] + features.tolist()
            processed_data.append(new_row)
            match_count += 1

# 4. Save
if match_count == 0:
    print("\nCRITICAL ERROR: Found the label column, but no gestures matched!")
    print(f"Labels found in file: {df[label_col_name].unique()}")
    print("Please update LABEL_MAP in the script to match these names.")
    exit()

# 63 cols (21 landmarks * 3 coords) + 1 label
column_names = ['label'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
final_df = pd.DataFrame(processed_data, columns=column_names)

output_file = 'gestures.csv'
final_df.to_csv(output_file, index=False)

print(f"\nSUCCESS!")
print(f"Processed {len(final_df)} samples.")
print(f"Saved to '{output_file}'.")
print("\nClass Distribution:")
print(final_df['label'].value_counts().sort_index())
