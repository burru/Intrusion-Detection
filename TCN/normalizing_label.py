import pandas as pd
import json

# ------------------------
# UNIFIED LABEL MAPPING
# ------------------------
unify_map = {
    # Benign
    'BENIGN': 'Benign',
    'Benign': 'Benign',
    'Normal Traffic': 'Benign',

    # Bot
    'Bots': 'Bot',
    'BOT': 'Bot',

    # Brute Force
    'Brute Force': 'BruteForce',
    'BRUTEFORCE': 'BruteForce',

    # DoS Group (combine everything)
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

    # PortScan
    'Port Scanning': 'PortScan',
    'Portmap': 'PortScan',

    # Infiltration
    'Infiltration': 'Infiltration',

    # Web Attack
    'Web Attacks': 'WebAttack',
    'WEB ATTACK': 'WebAttack',

    # Anything unknown fallback
}

fallback_label = "Other"

def unify_labels(input_csv, output_csv):
    print(f"\n[INFO] Processing {input_csv}")
    df = pd.read_csv(input_csv)

    # Apply mapping
    df['UnifiedLabel'] = df['Label'].map(unify_map).fillna(fallback_label)

    # Assign new encoded IDs
    df['UnifiedLabel_id'] = df['UnifiedLabel'].astype('category').cat.codes

    # Save new file
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved unified dataset to {output_csv}")

    # Save unified mapping
    mapping = {
        str(code): label
        for code, label in enumerate(df['UnifiedLabel'].astype('category').cat.categories)
    }
    json_path = output_csv.replace(".csv", "_UnifiedLabelMapping.json")
    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"[INFO] Saved mapping to {json_path}")


# --------------------------------------------------------
# Run for all cleaned files (edit filenames as needed)
# --------------------------------------------------------
files = [
    ("CICIDS2017-cleaned-for-tcn.csv", "CICIDS2017-cleaned-tcn.csv"),
    ("CICIDS2018-cleaned-for-tcn.csv", "CICIDS2018-cleaned-tcn.csv"),
    ("CICIDS2019-cleaned-for-tcn.csv", "CICIDS2019-cleaned-tcn.csv"),
]

for in_f, out_f in files:
    unify_labels(in_f, out_f)
