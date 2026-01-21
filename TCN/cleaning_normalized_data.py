import sys
import json
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    'Flow Duration', 'Tot Fwd Pkts', 'TotLen Fwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
    'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Bwd IAT Min', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
    'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
    'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'PSH Flag Cnt',
    'ACK Flag Cnt', 'Pkt Size Avg', 'Subflow Fwd Byts', 'Init Fwd Win Byts',
    'Init Bwd Win Byts', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Max',
    'Idle Min'
]
TARGET_COL = "Label"
ID_COL = "Label_id"

# map raw labels -> unified labels
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
    'UDPLag': 'DoS',
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

def clean_for_tcn(input_path: str, output_path: str):
    print(f"[INFO] Loading {input_path}")
    df = pd.read_csv(input_path)

    # drop any existing id
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {input_path}: {missing}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} column not found in {input_path}")

    # keep only features + Label
    df = df[FEATURE_COLUMNS + [TARGET_COL]]

    # numeric features
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    print(f"[INFO] Dropped {before - len(df)} rows with NaN/inf")

    # unify labels
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    df[TARGET_COL] = df[TARGET_COL].replace(UNIFY_MAP)

    # anything unknown -> Other
    df[TARGET_COL] = df[TARGET_COL].where(
        df[TARGET_COL].isin(LABEL_ORDER), other='Other'
    )

    # *** FIXED PART: use global dict, not cat.codes ***
    df[ID_COL] = df[TARGET_COL].map(label_to_id)

    # final column order
    df = df[FEATURE_COLUMNS + [TARGET_COL, ID_COL]]

    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved cleaned dataset to {output_path}")

    # same mapping for EVERY file
    mapping = {str(v): k for k, v in label_to_id.items()}
    map_path = output_path.replace(".csv", "_label_mapping.json")
    with open(map_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[INFO] Saved label mapping to {map_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_data_for_tcn.py input.csv output_clean.csv")
        sys.exit(1)

    clean_for_tcn(sys.argv[1], sys.argv[2])
