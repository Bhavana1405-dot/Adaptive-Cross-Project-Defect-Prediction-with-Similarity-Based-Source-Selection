import glob
import pandas as pd

# Use glob to get all file paths
files = glob.glob("data/*.csv")

# Load all files into a list of DataFrames
# This is safe because Pandas handles different row counts automatically
df_list = [pd.read_csv(f) for f in files]

# Combine them vertically
# This aligns everything by the 'classname' and other headers
combined_df = pd.concat(df_list, axis=0, ignore_index=True)

# Verify the result
print(f"Combined Shape: {combined_df.shape}")

# If you need a NumPy array for Machine Learning:
# Only convert the numeric columns to avoid 'Object' type issues
numeric_cols = combined_df.select_dtypes(include=['number'])
X = numeric_cols.drop(columns=['bug']).values
y = numeric_cols['bug'].values