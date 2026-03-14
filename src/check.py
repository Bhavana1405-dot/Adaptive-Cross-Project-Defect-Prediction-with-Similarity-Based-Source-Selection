import pandas as pd
import os

# Your specific column schema (24 columns total)
EXPECTED_COLUMNS = [
    "name", "version", "classname", "wmc", "dit", "noc", "cbo", "rfc", 
    "lcom", "ca", "ce", "npm", "lcom3", "loc", "dam", "moa", 
    "mfa", "cam", "ic", "cbm", "amc", "max_cc", "avg_cc", "bug"
]

def find_mismatched_csvs(folder_path='data'):
    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' not found.")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    problematic_files = 0

    print(f"Checking {len(csv_files)} files in '{folder_path}'...\n")

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            # Read the file
            df = pd.read_csv(file_path)
            actual_cols = list(df.columns)
            
            # Check 1: Column Count
            if len(actual_cols) != len(EXPECTED_COLUMNS):
                print(f"❌ {file}: Column count mismatch!")
                print(f"   -> Expected {len(EXPECTED_COLUMNS)}, but found {len(actual_cols)}.")
                problematic_files += 1
                continue

            # Check 2: Exact Order and Naming
            if actual_cols != EXPECTED_COLUMNS:
                print(f"⚠️ {file}: Columns out of order or renamed.")
                # Find the first mismatch for brevity
                for i, (a, b) in enumerate(zip(actual_cols, EXPECTED_COLUMNS)):
                    if a != b:
                        print(f"   -> Mismatch at index {i}: Expected '{b}', got '{a}'")
                        break
                problematic_files += 1

        except Exception as e:
            print(f"🔥 {file}: Critical error reading file: {e}")
            problematic_files += 1

    if problematic_files == 0:
        print("✅ Success: All CSV files in 'data/' match the required schema.")
    else:
        print(f"\nFound {problematic_files} files that need fixing.")

# Run the check
find_mismatched_csvs()