import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"

# These are the 21 numeric metrics shared across your datasets
# We exclude 'name', 'version', 'classname' (strings) and 'bug' (target)
NUMERIC_METRICS = [
    "wmc", "dit", "noc", "cbo", "rfc", "lcom", "ca", "ce", "npm", 
    "lcom3", "loc", "dam", "moa", "mfa", "cam", "ic", "cbm", 
    "amc", "max_cc", "avg_cc"
]

def get_project_vector(file_path):
    data = pd.read_csv(file_path)
    
    # Force selection of ONLY the specific numeric metrics 
    # This ensures every file returns a vector of length 20
    # even if some files have extra or missing columns
    available_metrics = [m for m in NUMERIC_METRICS if m in data.columns]
    
    X = data[available_metrics]
    
    # Fill missing columns with 0 if a specific file is missing a metric
    if len(available_metrics) < len(NUMERIC_METRICS):
        for missing in set(NUMERIC_METRICS) - set(available_metrics):
            X[missing] = 0
            
    # Ensure column order is identical for every single file
    X = X[NUMERIC_METRICS]

    return np.mean(X.values, axis=0)

# Collect all CSV files in data/
project_paths = []
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"The directory {DATA_DIR} does not exist.")

for file in os.listdir(DATA_DIR):
    full_path = os.path.join(DATA_DIR, file)
    if file.endswith(".csv") and os.path.isfile(full_path):
        project_paths.append(full_path)

project_vectors = {}
project_names = []

# Fill the dictionary
for path in project_paths:
    file_name = os.path.basename(path)
    try:
        vector = get_project_vector(path)
        project_names.append(file_name)
        project_vectors[file_name] = vector
    except Exception as e:
        print(f"Skipping {file_name} due to error: {e}")

# Convert to matrix - this will now work because all vectors are same length
vectors = np.array([project_vectors[name] for name in project_names])

# Normalize project representations
scaler = StandardScaler()
vectors = scaler.fit_transform(vectors)

# Choose target project - ensure the name matches your file exactly
target_project = "ant.csv" # Changed to match common naming like your output

if target_project not in project_names:
    print(f"Error: {target_project} not found. Available: {project_names[:3]}...")
else:
    target_index = project_names.index(target_project)
    target_vector = vectors[target_index].reshape(1, -1)

    # Compute similarity
    similarities = cosine_similarity(target_vector, vectors)[0]

    # Rank projects
    sorted_indices = np.argsort(similarities)[::-1]

    print("\nSimilarity Ranking (for target:", target_project, ")")
    for idx in sorted_indices:
        # Avoid showing the target project as its own most similar
        if project_names[idx] == target_project:
            continue
        print(f"{project_names[idx]:<20} -> {round(similarities[idx], 4)}")