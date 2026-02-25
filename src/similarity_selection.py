import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = "data"

def get_project_vector(file_path):
    data = pd.read_csv(file_path)

    X = data.drop("bug", axis=1)
    X = X.select_dtypes(include=["number"])

    return np.mean(X.values, axis=0)

# Collect all CSV files in data/
project_paths = []

for file in os.listdir(DATA_DIR):
    full_path = os.path.join(DATA_DIR, file)
    if file.endswith(".csv") and os.path.isfile(full_path):
        project_paths.append(full_path)

project_vectors = {}
project_names = []

# 🔥 Fill the dictionary properly
for path in project_paths:
    file_name = os.path.basename(path)
    project_names.append(file_name)
    project_vectors[file_name] = get_project_vector(path)

# Convert to matrix
vectors = np.array(list(project_vectors.values()))

# Normalize project representations
scaler = StandardScaler()
vectors = scaler.fit_transform(vectors)

# Choose target project
target_project = "camel.csv"

if target_project not in project_names:
    raise ValueError(f"{target_project} not found in data folder!")

target_index = project_names.index(target_project)
target_vector = vectors[target_index].reshape(1, -1)

# Compute similarity
similarities = cosine_similarity(target_vector, vectors)[0]

# Rank projects
sorted_indices = np.argsort(similarities)[::-1]

print("\nSimilarity Ranking (for target:", target_project, ")")
for idx in sorted_indices:
    print(project_names[idx], "->", round(similarities[idx], 4))