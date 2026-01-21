import pandas as pd

# Load your dataset
df = pd.read_csv("CICIDS2019-tcn.csv")

# Columns you want to remove
cols_to_drop = [
    "Dst Port",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Subflow Fwd Byts",
    "Active Mean", "Active Max", "Active Min",
    "Idle Mean", "Idle Max", "Idle Min"
]

# Drop the columns ONLY if they exist in the dataset
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Save cleaned dataset
df.to_csv("CICIDS2019-tcn.csv", index=False)

print("Columns dropped successfully!")
