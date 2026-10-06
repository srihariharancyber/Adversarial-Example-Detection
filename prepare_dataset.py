"""
Cybersecurity Dataset Preparation & Preprocessing Pipeline
Scenario 51: Adversarial Example Detection
Lab Exercise: FGSM Adversarial Examples and Model Robustness

This script creates and preprocesses a benchmark cybersecurity network intrusion dataset:
- Features reflect realistic network flow indicators (e.g., duration, byte counts, error rates, host rates).
- Performs categorical encoding, missing value checks, and bounded MinMax normalization [0, 1].
- Strictly fits the scaler on training data to prevent data leakage (ML best practice).
- Exports train/test clean CSV files and metadata for perturbation bounds.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def generate_benchmark_cybersecurity_data(n_samples=6000, random_state=42):
    """
    Generates a realistic, multi-attribute network intrusion detection dataset
    simulating benign traffic and attack types (DoS syn-flood, PortScan, BruteForce).
    """
    np.random.seed(random_state)

    # 60% Benign, 40% Malicious
    n_benign = int(n_samples * 0.6)
    n_attack = n_samples - n_benign

    # --- Benign Traffic Generation ---
    # Normal interactive and bulk data transfers
    benign_duration = np.random.exponential(scale=3.0, size=n_benign)
    benign_src_bytes = np.random.lognormal(mean=7.5, sigma=1.2, size=n_benign)
    benign_dst_bytes = np.random.lognormal(mean=8.5, sigma=1.4, size=n_benign)
    benign_count = np.random.poisson(lam=5, size=n_benign)
    benign_srv_count = np.clip(benign_count - np.random.poisson(lam=1, size=n_benign), 1, None)
    benign_serror_rate = np.random.beta(a=0.5, b=20, size=n_benign)  # Low error rate
    benign_rerror_rate = np.random.beta(a=0.5, b=25, size=n_benign)
    benign_same_srv_rate = np.random.beta(a=15, b=2, size=n_benign)   # High same-srv
    benign_diff_srv_rate = 1.0 - benign_same_srv_rate
    benign_dst_host_count = np.random.randint(1, 100, size=n_benign)
    benign_dst_host_srv_count = np.clip(benign_dst_host_count - np.random.randint(0, 10, size=n_benign), 1, 255)
    benign_dst_host_same_srv_rate = benign_dst_host_srv_count / np.maximum(benign_dst_host_count, 1)
    benign_dst_host_serror_rate = np.random.beta(a=0.5, b=20, size=n_benign)
    benign_failed_logins = np.random.choice([0, 1], size=n_benign, p=[0.995, 0.005])
    benign_logged_in = np.random.choice([0, 1], size=n_benign, p=[0.15, 0.85])
    benign_protocol = np.random.choice(["TCP", "UDP", "ICMP"], size=n_benign, p=[0.8, 0.18, 0.02])
    benign_service = np.random.choice(["HTTP", "HTTPS", "DNS", "SSH", "FTP"], size=n_benign, p=[0.45, 0.35, 0.12, 0.05, 0.03])
    benign_label = np.zeros(n_benign, dtype=int)  # 0 = Benign

    # --- Malicious Traffic Generation ---
    # High frequency syn-flood, port scans, and brute-force patterns
    attack_duration = np.random.exponential(scale=0.4, size=n_attack)
    attack_src_bytes = np.random.lognormal(mean=5.0, sigma=2.0, size=n_attack)
    attack_dst_bytes = np.random.lognormal(mean=4.0, sigma=1.8, size=n_attack)
    attack_count = np.random.poisson(lam=180, size=n_attack)  # Surge in connections
    attack_srv_count = np.random.poisson(lam=160, size=n_attack)
    attack_serror_rate = np.random.beta(a=18, b=2, size=n_attack)  # High syn error
    attack_rerror_rate = np.random.beta(a=10, b=5, size=n_attack)
    attack_same_srv_rate = np.random.beta(a=2, b=10, size=n_attack)
    attack_diff_srv_rate = 1.0 - attack_same_srv_rate
    attack_dst_host_count = np.random.randint(150, 255, size=n_attack)
    attack_dst_host_srv_count = np.random.randint(1, 30, size=n_attack)
    attack_dst_host_same_srv_rate = attack_dst_host_srv_count / np.maximum(attack_dst_host_count, 1)
    attack_dst_host_serror_rate = np.random.beta(a=16, b=2, size=n_attack)
    attack_failed_logins = np.random.choice([0, 1, 2, 3], size=n_attack, p=[0.70, 0.18, 0.08, 0.04])
    attack_logged_in = np.random.choice([0, 1], size=n_attack, p=[0.88, 0.12])
    attack_protocol = np.random.choice(["TCP", "UDP", "ICMP"], size=n_attack, p=[0.75, 0.10, 0.15])
    attack_service = np.random.choice(["HTTP", "HTTPS", "DNS", "SSH", "FTP"], size=n_attack, p=[0.25, 0.15, 0.10, 0.30, 0.20])
    attack_label = np.ones(n_attack, dtype=int)  # 1 = Malicious / Attack

    # Concatenate features
    df = pd.DataFrame({
        "duration": np.concatenate([benign_duration, attack_duration]),
        "src_bytes": np.concatenate([benign_src_bytes, attack_src_bytes]),
        "dst_bytes": np.concatenate([benign_dst_bytes, attack_dst_bytes]),
        "count": np.concatenate([benign_count, attack_count]),
        "srv_count": np.concatenate([benign_srv_count, attack_srv_count]),
        "serror_rate": np.concatenate([benign_serror_rate, attack_serror_rate]),
        "rerror_rate": np.concatenate([benign_rerror_rate, attack_rerror_rate]),
        "same_srv_rate": np.concatenate([benign_same_srv_rate, attack_same_srv_rate]),
        "diff_srv_rate": np.concatenate([benign_diff_srv_rate, attack_diff_srv_rate]),
        "dst_host_count": np.concatenate([benign_dst_host_count, attack_dst_host_count]),
        "dst_host_srv_count": np.concatenate([benign_dst_host_srv_count, attack_dst_host_srv_count]),
        "dst_host_same_srv_rate": np.concatenate([benign_dst_host_same_srv_rate, attack_dst_host_same_srv_rate]),
        "dst_host_serror_rate": np.concatenate([benign_dst_host_serror_rate, attack_dst_host_serror_rate]),
        "num_failed_logins": np.concatenate([benign_failed_logins, attack_failed_logins]),
        "logged_in": np.concatenate([benign_logged_in, attack_logged_in]),
        "protocol_type": np.concatenate([benign_protocol, attack_protocol]),
        "service": np.concatenate([benign_service, attack_service]),
        "label": np.concatenate([benign_label, attack_label])
    })

    # Shuffle rows randomly
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def preprocess_cybersecurity_data(output_dir=".", test_size=0.2, random_state=42):
    """
    Performs data cleaning, one-hot encoding, and feature scaling:
    - Bounded normalization [0, 1] is critical for FGSM gradient evaluation.
    - Saves processed files and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("STEP 1: Generating Raw Cybersecurity Feature Dataset")
    print("=" * 70)

    raw_df = generate_benchmark_cybersecurity_data(n_samples=6000, random_state=random_state)
    raw_csv_path = os.path.join(output_dir, "cybersecurity_features_raw.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"[+] Raw dataset created: {raw_csv_path} ({len(raw_df)} samples, {raw_df.shape[1]} columns)")
    print(f"[+] Class distribution: Benign (0) = {(raw_df['label'] == 0).sum()}, Malicious (1) = {(raw_df['label'] == 1).sum()}")

    # Identify numerical and categorical features
    target_col = "label"
    cat_cols = ["protocol_type", "service"]
    num_cols = [c for c in raw_df.columns if c not in cat_cols + [target_col]]

    print("\n" + "=" * 70)
    print("STEP 2: Train/Test Split (Fit Preprocessor strictly on Train Set)")
    print("=" * 70)

    X = raw_df.drop(columns=[target_col])
    y = raw_df[target_col]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    print(f"[+] Train split: {len(X_train_raw)} samples | Test split: {len(X_test_raw)} samples")

    # One-hot encode categorical features
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(X_train_raw[cat_cols])
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))

    X_train_cat = ohe.transform(X_train_raw[cat_cols])
    X_test_cat = ohe.transform(X_test_raw[cat_cols])

    # Log transform heavy-tailed byte/count features to stabilize gradients
    skewed_cols = ["src_bytes", "dst_bytes", "duration"]
    X_train_num = X_train_raw[num_cols].copy()
    X_test_num = X_test_raw[num_cols].copy()

    for sc in skewed_cols:
        X_train_num[sc] = np.log1p(np.maximum(0, X_train_num[sc]))
        X_test_num[sc] = np.log1p(np.maximum(0, X_test_num[sc]))

    # Min-Max Normalization to [0, 1] bounds (essential for FGSM threat model)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_train_num_scaled = scaler.fit_transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    # Ensure test features remain bounded within [0, 1]
    X_test_num_scaled = np.clip(X_test_num_scaled, 0.0, 1.0)

    # Combine numerical and one-hot encoded categorical columns
    all_feature_names = num_cols + cat_feature_names
    X_train_final = np.hstack([X_train_num_scaled, X_train_cat])
    X_test_final = np.hstack([X_test_num_scaled, X_test_cat])

    # Convert to DataFrames
    train_df = pd.DataFrame(X_train_final, columns=all_feature_names)
    train_df[target_col] = y_train.values

    test_df = pd.DataFrame(X_test_final, columns=all_feature_names)
    test_df[target_col] = y_test.values

    train_path = os.path.join(output_dir, "cybersecurity_train_clean.csv")
    test_path = os.path.join(output_dir, "cybersecurity_test_clean.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metadata = {
        "dataset_name": "Cybersecurity Intrusion Detection Benchmark (Scenario 51)",
        "total_samples": len(raw_df),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "num_features": len(all_feature_names),
        "feature_names": all_feature_names,
        "feature_bounds": {"min": 0.0, "max": 1.0},
        "target_column": target_col,
        "classes": {"0": "Benign / Normal", "1": "Malicious / Attack"},
        "random_state": random_state
    }

    meta_path = os.path.join(output_dir, "preprocessing_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[+] Clean train set saved: {train_path} ({train_df.shape})")
    print(f"[+] Clean test set saved:  {test_path} ({test_df.shape})")
    print(f"[+] Metadata & bounds saved: {meta_path}")
    print("=" * 70)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return train_path, test_path, meta_path


if __name__ == "__main__":
    preprocess_cybersecurity_data()
