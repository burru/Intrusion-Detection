"""
clean_42_shared_features.py

Usage:
    python clean_42_shared_features.py input.csv output_clean.csv

What it does:
    - Keeps only the 42 shared numeric flow features across all 3 datasets
    - Converts features to numeric, removes inf / NaN rows
    - Unifies raw labels into global classes
    - Assigns consistent Label_id values (0..7) for all datasets
    - Saves cleaned CSV and a JSON label mapping (same for every file)
"""

import sys
import json
import numpy as np
import pandas as pd

# 42 SHARED FEATURE COLUMNS (after dropping the 11 extra ones)
FEATURE_COLUMNS = [
    'Flow Duration', 'Tot Fwd Pkts', 'TotLen Fwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
    'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
    'Flow Pkts/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd Header Len',
    'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s', 'Pkt Len Min',
    'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std', 'Pkt Len Var',
    'FIN Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'Pkt Size Avg'
]

TARGET_COL = "Label"
ID_COL = "Label_id"

# ---------- RAW LABEL -> UNIFIED LABEL MAPPING ----------
UNIFY_MAP = {
    # Benign
    'BENIGN': 'Benign',
    'Benign': 'Benign',
    'Normal Traffic': 'Benign',

    # Bot
    'Bots': 'Bot',
    'BOT': 'Bot',

    # Brute force
    'Brute Force': 'BruteForce',
    'BRUTEFORCE': 'BruteForce',

    # DoS / DDoS & variants
    'DoS': 'DoS',
    'DDoS': 'DoS',
    'DrDoS_DNS': 'DoS',
    'DrDoS_LDAP': 'DoS',
    'DrDoS_MSSQL': 'DoS',
    'DrDoS_NTP': 'DoS',
    'DrDoS_NetBIOS': 'DoS',
    'DrDoS_SNMP': 'DoS',
    'DrDoS_UDP': 'DoS',
    'UDP': 'DoS',
    'UDP-lag': 'DoS',
    'UDP-Lag': 'DoS',
    'WebDDoS': 'DoS',
    'LDAP': 'DoS',
    'MSSQL': 'DoS',
    'NetBIOS': 'DoS',
    'Syn': 'DoS',
    'TFTP': 'DoS',

    # Port scanning
    'Port Scanning': 'PortScan',
    'Portmap': 'PortScan',

    # Infiltration
    'Infiltration': 'Infiltration',

    # Web attacks
    'Web Attacks': 'WebAttack',
    'WEB ATTACK': 'WebAttack',
}

LABEL_ORDER = [
    'Benign',
    'Bot',
    'BruteForce',
    'DoS',
    'PortScan',
    'Infiltration',
    'WebAttack',
    'Other'
]
label_to_id = {lab: i for i, lab in enumerate(LABEL_ORDER)}


def clean_dataset(input_path: str, output_path: str):
    print(f"[INFO] Loading {input_path}")
    df = pd.read_csv(input_path)

    # Drop any existing Label_id – we will recreate it
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    # Check columns
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in {input_path}: {missing_features}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} column not found in {input_path}")

    # Keep only 42 features + label
    df = df[FEATURE_COLUMNS + [TARGET_COL]]

    # Convert features to numeric, coerce errors to NaN
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    # Replace inf/-inf with NaN and drop any rows with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"[INFO] Dropped {before - len(df)} rows with NaN/inf")

    # ---------- unify labels ----------
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df[TARGET_COL] = df[TARGET_COL].replace(UNIFY_MAP)

    # Anything not in LABEL_ORDER goes to "Other"
    df[TARGET_COL] = df[TARGET_COL].where(
        df[TARGET_COL].isin(LABEL_ORDER), other='Other'
    )

    # Map to global IDs (0..7)
    df[ID_COL] = df[TARGET_COL].map(label_to_id)

    # Final column order: 42 features + Label + Label_id
    df = df[FEATURE_COLUMNS + [TARGET_COL, ID_COL]]

    # Save cleaned CSV
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved cleaned dataset to {output_path}")

    # Save *global* label mapping (same for all datasets)
    mapping = {str(v): k for k, v in label_to_id.items()}
    map_path = output_path.replace(".csv", "_label_mapping.json")
    with open(map_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[INFO] Saved label mapping to {map_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_42_shared_features.py input.csv output_clean.csv")
        sys.exit(1)

    in_csv = sys.argv[1]
    out_csv = sys.argv[2]
    clean_dataset(in_csv, out_csv)
