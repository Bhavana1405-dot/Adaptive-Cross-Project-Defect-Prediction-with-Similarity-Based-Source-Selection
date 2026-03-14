"""
Run this to find exactly where the pipeline is hanging.
Usage: python src_new/debug_run.py
"""
import sys, os

print("Step 1: Python is working", flush=True)

print("Step 2: Importing numpy/pandas...", flush=True)
import numpy as np
import pandas as pd
print("       OK", flush=True)

print("Step 3: Importing sklearn...", flush=True)
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
print("       OK", flush=True)

print("Step 4: Importing local modules...", flush=True)
sys.path.insert(0, os.path.dirname(__file__))

print("       source_selector...", flush=True)
from source_selector import SimilaritySourceSelector
print("       OK", flush=True)

print("       feature_selection...", flush=True)
from feature_selection import LASSOREFSelector
print("       OK", flush=True)

print("       feature_extraction...", flush=True)
from feature_extraction import FeatureExtractor
print("       OK", flush=True)

print("       coral...", flush=True)
from coral import CORAL
print("       OK", flush=True)

print("       smote...", flush=True)
from smote import SMOTE
print("       OK", flush=True)

print("       ensemble...", flush=True)
from ensemble import StackedEnsemble
print("       OK", flush=True)

print("\nStep 5: Listing data/ folder...", flush=True)
data_dir = "data/"
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
print(f"       Found {len(files)} CSV files: {files[:5]}{'...' if len(files)>5 else ''}", flush=True)

print("\nStep 6: Loading first CSV...", flush=True)
first_csv = os.path.join(data_dir, files[0])
df = pd.read_csv(first_csv)
print(f"       {files[0]}: {df.shape} — columns: {list(df.columns[:5])}...", flush=True)

print("\nStep 7: Running FeatureExtractor on first CSV...", flush=True)
ext = FeatureExtractor(mode='metrics')
X = ext.extract(first_csv, project_name=files[0].replace('.csv',''))
print(f"       Shape: {X.shape}", flush=True)

print("\nStep 8: Loading all projects...", flush=True)
from pipeline import load_all_projects
projects = load_all_projects(data_dir, feature_mode='metrics')
print(f"\nAll projects loaded:", flush=True)
for name, (X, y) in projects.items():
    print(f"  {name:30s} X={X.shape}  defect_rate={y.mean():.1%}", flush=True)

print("\n✅ Everything working. Ready to run pipeline.", flush=True)