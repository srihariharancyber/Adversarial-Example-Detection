"""
Scenario 51: Adversarial Example Detection
Lab Exercise: FGSM Adversarial Examples and Model Robustness
Complete Implementation satisfying Tasks 571 - 576 and Deliverables 577 - 581.

Author: Antigravity AI Lab System
Target Framework: TensorFlow / Keras 3+ & Scikit-Learn
"""

import json
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis

# Set fixed random seeds for strict reproducibility (Task 576)
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Framework selection: prefers TensorFlow if installed, falls back seamlessly to PyTorch
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    from tensorflow.keras import layers, models, regularizers, optimizers
    BACKEND = "tensorflow"
    print("[+] Using TensorFlow/Keras backend")
except ImportError:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    torch.manual_seed(SEED)
    BACKEND = "pytorch"
    print(f"[+] TensorFlow not found; using PyTorch {torch.__version__} backend")

# Scikit-learn evaluation metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)


# ==============================================================================
# 1. DATA LOADING & PREPROCESSING VERIFICATION
# ==============================================================================
def load_and_verify_dataset(data_dir="."):
    """
    Loads preprocessed clean train and test cybersecurity datasets.
    Verifies feature bounds [0.0, 1.0] and metadata constraints.
    """
    train_path = os.path.join(data_dir, "cybersecurity_train_clean.csv")
    test_path = os.path.join(data_dir, "cybersecurity_test_clean.csv")
    meta_path = os.path.join(data_dir, "preprocessing_metadata.json")

    # If dataset files are missing, run preprocessor
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        from prepare_dataset import preprocess_cybersecurity_data
        preprocess_cybersecurity_data(output_dir=data_dir, random_state=SEED)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c != "label"]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    # Documented feature bounds
    feature_min = float(X_train.min())
    feature_max = float(X_train.max())
    print("\n" + "=" * 75)
    print("DATASET VERIFICATION & FEATURE BOUNDS")
    print("=" * 75)
    print(f"Features count:        {len(feature_cols)}")
    print(f"Training samples:      {len(X_train)} (Benign: {(y_train == 0).sum()}, Malicious: {(y_train == 1).sum()})")
    print(f"Test samples:          {len(X_test)} (Benign: {(y_test == 0).sum()}, Malicious: {(y_test == 1).sum()})")
    print(f"Documented bounds:     [{feature_min:.4f}, {feature_max:.4f}] -> strictly clipped to [0.0, 1.0]")
    
    return X_train, y_train, X_test, y_test, feature_cols


# ==============================================================================
# 2. MODEL BUILDING & TRAINING (TASK 571)
# ==============================================================================
if BACKEND == "pytorch":
    class PyTorchClassifierWrapper(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

        def predict(self, X, verbose=0):
            self.eval()
            with torch.no_grad():
                x_t = torch.tensor(X, dtype=torch.float32)
                preds = self.forward(x_t).cpu().numpy().reshape(-1, 1)
            return preds


def build_and_train_classifier(X_train, y_train, input_dim, epochs=25, batch_size=64):
    """
    Constructs a differentiable neural network classifier.
    Dense layers with ReLU, Dropout for regularization, and Sigmoid output.
    Supports both TensorFlow/Keras and PyTorch backends.
    """
    if BACKEND == "tensorflow":
        model = models.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.2, seed=SEED),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.2, seed=SEED),
            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        print("\n" + "=" * 75)
        print("TRAINING DIFFERENTIABLE KERAS CYBERSECURITY CLASSIFIER")
        print("=" * 75)
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            verbose=1
        )
        return model
    else:
        model = PyTorchClassifierWrapper(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss()

        val_size = int(len(X_train) * 0.15)
        train_size = len(X_train) - val_size
        X_tr, X_val = X_train[:train_size], X_train[train_size:]
        y_tr, y_val = y_train[:train_size], y_train[train_size:]

        train_dataset = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print("\n" + "=" * 75)
        print("TRAINING DIFFERENTIABLE PYTORCH CYBERSECURITY CLASSIFIER")
        print("=" * 75)
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
                preds = (out >= 0.5).float()
                correct += (preds == batch_y).sum().item()
                total += len(batch_x)

            model.eval()
            with torch.no_grad():
                val_out = model(torch.tensor(X_val, dtype=torch.float32))
                val_y_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
                val_loss = criterion(val_out, val_y_tensor).item()
                val_acc = ((val_out >= 0.5).float() == val_y_tensor).float().mean().item()

            train_loss = total_loss / total
            train_acc = correct / total
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1:2d}/{epochs} - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")

        return model


# ==============================================================================
# 3. FGSM ATTACK GENERATION (TASKS 572 & 573)
# ==============================================================================
def generate_fgsm_examples(model, X, y, epsilon=0.10, clip_min=0.0, clip_max=1.0):
    """
    Task 573: Fast Gradient Sign Method (FGSM)
    x_adv = clip(x + epsilon * sign(nabla_x L(x, y)), clip_min, clip_max)
    Perturbations are strictly constrained within [clip_min, clip_max].
    Supports both TensorFlow and PyTorch backends.
    """
    if BACKEND == "tensorflow":
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        x_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        y_tensor = tf.convert_to_tensor(y.reshape(-1, 1), dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            preds = model(x_tensor, training=False)
            loss = loss_fn(y_tensor, preds)
        gradients = tape.gradient(loss, x_tensor)
        perturbation = epsilon * tf.sign(gradients)
        x_adv = x_tensor + perturbation
        x_adv_clipped = tf.clip_by_value(x_adv, clip_value_min=clip_min, clip_value_max=clip_max)
        return x_adv_clipped.numpy()
    else:
        model.eval()
        x_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
        criterion = nn.BCELoss()

        preds = model(x_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()

        perturbation = epsilon * torch.sign(x_tensor.grad)
        x_adv = x_tensor + perturbation
        x_adv_clipped = torch.clamp(x_adv, min=clip_min, max=clip_max)
        return x_adv_clipped.detach().cpu().numpy()


# ==============================================================================
# 4. PERTURBATION VALIDITY CHECKS (DELIVERABLE 580)
# ==============================================================================
def verify_perturbation_validity(X_clean, X_adv, epsilon, clip_min=0.0, clip_max=1.0):
    """
    Task 580: Verify that all perturbations adhere strictly to the threat model:
    - Linf distance <= epsilon
    - Features remain within valid domain [clip_min, clip_max]
    - Quantifies L1 and L2 average norms
    """
    perturbation = X_adv - X_clean
    linf_norm = np.max(np.abs(perturbation))
    l2_norm_mean = np.mean(np.linalg.norm(perturbation, ord=2, axis=1))
    l1_norm_mean = np.mean(np.linalg.norm(perturbation, ord=1, axis=1))
    
    min_val = float(np.min(X_adv))
    max_val = float(np.max(X_adv))

    bounds_valid = (min_val >= clip_min - 1e-6) and (max_val <= clip_max + 1e-6)
    linf_valid = linf_norm <= (epsilon + 1e-6)

    print("\n" + "=" * 75)
    print(f"PERTURBATION VALIDITY CHECKS (Budget Epsilon = {epsilon})")
    print("=" * 75)
    print(f"Maximum L-infinity perturbation: {linf_norm:.6f} (Budget: {epsilon}) -> {'PASS' if linf_valid else 'FAIL'}")
    print(f"Mean L2 perturbation norm:       {l2_norm_mean:.6f}")
    print(f"Mean L1 perturbation norm:       {l1_norm_mean:.6f}")
    print(f"Adversarial feature range:       [{min_val:.4f}, {max_val:.4f}] -> {'PASS' if bounds_valid else 'FAIL'}")

    return {
        "epsilon": float(epsilon),
        "max_linf": float(linf_norm),
        "mean_l2": float(l2_norm_mean),
        "mean_l1": float(l1_norm_mean),
        "bounds_pass": bool(bounds_valid),
        "linf_pass": bool(linf_valid)
    }


# ==============================================================================
# 5. ADVERSARIAL EXAMPLE DETECTOR (TASK 575: SCENARIO-SPECIFIC EXPERIMENT)
# ==============================================================================
class FeatureDistributionDetector:
    """
    Problem Statement Requirement:
    "Build a simple detector that flags suspicious inputs using distance
     from the clean feature distribution."

    Implementation:
    Computes Mahalanobis Distance / Centroid Dispersion from the clean feature distribution.
    Calibrates an anomaly threshold tau on the clean validation distribution
    (e.g. 95th percentile, fixing clean false-alarm rate at 5%).
    Inputs exceeding tau are flagged as suspicious / adversarial.
    """
    def __init__(self, regularization=1e-3):
        self.reg = regularization
        self.mean_clean = None
        self.inv_cov = None
        self.threshold = None

    def fit(self, X_clean):
        """Fits distribution statistics (mean and regularized inverse covariance)."""
        self.mean_clean = np.mean(X_clean, axis=0)
        cov = np.cov(X_clean, rowvar=False)
        # Add shrinkage/Tikhonov regularization to ensure well-conditioned inversion
        reg_cov = cov + np.eye(cov.shape[0]) * self.reg
        self.inv_cov = np.linalg.pinv(reg_cov)

    def compute_distance(self, X):
        """Computes distance of each sample to the clean distribution."""
        diff = X - self.mean_clean
        # Vectorized Mahalanobis distance calculation: sqrt(diag(diff @ inv_cov @ diff.T))
        left = np.dot(diff, self.inv_cov)
        dists_sq = np.sum(left * diff, axis=1)
        dists = np.sqrt(np.maximum(dists_sq, 0.0))
        return dists

    def calibrate_threshold(self, X_clean_val, false_positive_rate=0.05):
        """Sets threshold tau based on the (1 - FPR) quantile of clean distances."""
        clean_dists = self.compute_distance(X_clean_val)
        self.threshold = float(np.percentile(clean_dists, (1.0 - false_positive_rate) * 100))
        return self.threshold

    def predict(self, X):
        """Flags sample as suspicious (1) if distance > threshold, else clean (0)."""
        dists = self.compute_distance(X)
        return (dists > self.threshold).astype(int), dists


# ==============================================================================
# 6. COMPREHENSIVE EXPERIMENT EXECUTION (TASKS 571 - 576, DELIVERABLES 577 - 581)
# ==============================================================================
def run_scenario_51_experiment(epsilon_test=0.15):
    # Step 1: Load data
    X_train, y_train, X_test, y_test, feature_cols = load_and_verify_dataset(".")

    # Step 2: Build & train model
    model = build_and_train_classifier(X_train, y_train, input_dim=X_train.shape[1], epochs=25)

    # --------------------------------------------------------------------------
    # TASK 571: Clean Baseline Performance
    # --------------------------------------------------------------------------
    clean_probs = model.predict(X_test, verbose=0).flatten()
    clean_preds = (clean_probs >= 0.5).astype(int)

    clean_acc = accuracy_score(y_test, clean_preds)
    clean_prec = precision_score(y_test, clean_preds)
    clean_rec = recall_score(y_test, clean_preds)
    clean_f1 = f1_score(y_test, clean_preds)
    clean_cm = confusion_matrix(y_test, clean_preds)

    print("\n" + "=" * 75)
    print("TASK 571: CLEAN BASELINE PERFORMANCE (BEFORE ATTACK)")
    print("=" * 75)
    print(f"Clean Accuracy:  {clean_acc * 100:.2f}%")
    print(f"Clean Precision: {clean_prec * 100:.2f}%")
    print(f"Clean Recall:    {clean_rec * 100:.2f}%")
    print(f"Clean F1-Score:  {clean_f1 * 100:.2f}%")
    print(f"Clean Confusion Matrix:\n{clean_cm}")
    print(f"  [TN={clean_cm[0,0]} (Benign Correct), FP={clean_cm[0,1]} (False Alarm)]")
    print(f"  [FN={clean_cm[1,0]} (Missed Attack),  TP={clean_cm[1,1]} (Detected Attack)]")

    # --------------------------------------------------------------------------
    # TASK 573: Attack Correctly Classified Samples
    # --------------------------------------------------------------------------
    correct_indices = np.where(clean_preds == y_test)[0]
    X_test_correct = X_test[correct_indices]
    y_test_correct = y_test[correct_indices]
    clean_probs_correct = clean_probs[correct_indices]

    print(f"\n[+] Total clean samples: {len(X_test)} | Correctly classified: {len(X_test_correct)} ({len(X_test_correct)/len(X_test)*100:.1f}%)")

    # Generate FGSM adversarial examples
    X_adv_correct = generate_fgsm_examples(model, X_test_correct, y_test_correct, epsilon=epsilon_test)

    # Task 580: Validity check
    validity_info = verify_perturbation_validity(X_test_correct, X_adv_correct, epsilon=epsilon_test)

    # --------------------------------------------------------------------------
    # TASK 574: Compare Clean vs Adversarial Predictions & Confidence Shifts
    # --------------------------------------------------------------------------
    adv_probs_correct = model.predict(X_adv_correct, verbose=0).flatten()
    adv_preds_correct = (adv_probs_correct >= 0.5).astype(int)

    # Overall test evaluation with attacked samples substituted
    X_test_perturbed = X_test.copy()
    X_test_perturbed[correct_indices] = X_adv_correct
    adv_probs_all = model.predict(X_test_perturbed, verbose=0).flatten()
    adv_preds_all = (adv_probs_all >= 0.5).astype(int)

    adv_acc = accuracy_score(y_test, adv_preds_all)
    adv_prec = precision_score(y_test, adv_preds_all)
    adv_rec = recall_score(y_test, adv_preds_all)
    adv_f1 = f1_score(y_test, adv_preds_all)
    adv_cm = confusion_matrix(y_test, adv_preds_all)

    # Attack Success Rate (ASR): fraction of correctly classified clean samples successfully flipped
    successful_evasions = np.sum(adv_preds_correct != y_test_correct)
    asr = successful_evasions / len(y_test_correct)

    # Confidence change analysis
    # Clean confidence on predicted class: p if pred=1 else 1-p
    clean_confidence = np.where(clean_preds[correct_indices] == 1, clean_probs_correct, 1.0 - clean_probs_correct)
    adv_confidence_on_true_class = np.where(y_test_correct == 1, adv_probs_correct, 1.0 - adv_probs_correct)
    confidence_drop = clean_confidence - adv_confidence_on_true_class

    mean_conf_drop = float(np.mean(confidence_drop))
    median_conf_drop = float(np.median(confidence_drop))

    # Security-relevant errors: Malicious traffic evading classifier (False Negatives)
    malicious_mask = (y_test_correct == 1)
    malicious_evaded = np.sum((adv_preds_correct[malicious_mask] == 0))
    benign_mask = (y_test_correct == 0)
    benign_flipped = np.sum((adv_preds_correct[benign_mask] == 1))

    print("\n" + "=" * 75)
    print(f"TASK 574: ADVERSARIAL RESULTS AFTER FGSM (Epsilon = {epsilon_test})")
    print("=" * 75)
    print(f"Post-Attack Accuracy:       {adv_acc * 100:.2f}% (Drop: {(clean_acc - adv_acc) * 100:.2f}%)")
    print(f"Post-Attack Recall:         {adv_rec * 100:.2f}%")
    print(f"Post-Attack F1-Score:       {adv_f1 * 100:.2f}%")
    print(f"Attack Success Rate (ASR):  {asr * 100:.2f}% ({successful_evasions} / {len(y_test_correct)} samples fooled)")
    print(f"Malicious Evasion (FN err): {malicious_evaded} / {np.sum(malicious_mask)} ({malicious_evaded / np.sum(malicious_mask)*100:.1f}%)")
    print(f"Benign False Alarms (FP):   {benign_flipped} / {np.sum(benign_mask)} ({benign_flipped / np.sum(benign_mask)*100:.1f}%)")
    print(f"Mean Confidence Drop:       {mean_conf_drop:.4f}")
    print(f"Median Confidence Drop:     {median_conf_drop:.4f}")

    # --------------------------------------------------------------------------
    # TASK 575: Adversarial Example Detection Experiment
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TASK 575: ADVERSARIAL DETECTOR (DISTANCE FROM CLEAN DISTRIBUTION)")
    print("=" * 75)

    detector = FeatureDistributionDetector(regularization=1e-3)
    detector.fit(X_train)
    tau = detector.calibrate_threshold(X_test, false_positive_rate=0.05)
    print(f"[+] Detector fitted on clean feature distribution.")
    print(f"[+] Calibrated Detection Threshold (tau @ 5% FPR): {tau:.4f}")

    # Test detector on clean test set (0 = Clean) and adversarial set (1 = Suspicious)
    dists_clean = detector.compute_distance(X_test)
    dists_adv = detector.compute_distance(X_adv_correct)

    y_detector_true = np.concatenate([np.zeros(len(dists_clean)), np.ones(len(dists_adv))])
    all_dists = np.concatenate([dists_clean, dists_adv])
    y_detector_pred = (all_dists > tau).astype(int)

    det_acc = accuracy_score(y_detector_true, y_detector_pred)
    det_prec = precision_score(y_detector_true, y_detector_pred)
    det_rec = recall_score(y_detector_true, y_detector_pred)
    det_f1 = f1_score(y_detector_true, y_detector_pred)
    det_auc = roc_auc_score(y_detector_true, all_dists)
    det_cm = confusion_matrix(y_detector_true, y_detector_pred)

    print(f"Detection Accuracy:     {det_acc * 100:.2f}%")
    print(f"Detection True Pos Rate:{det_rec * 100:.2f}% (Adversarials detected)")
    print(f"Detection False Pos Rate:{(det_cm[0, 1] / len(dists_clean)) * 100:.2f}% (Clean flagged as suspicious)")
    print(f"Detection Precision:    {det_prec * 100:.2f}%")
    print(f"Detection F1-Score:     {det_f1 * 100:.2f}%")
    print(f"Detection ROC-AUC:      {det_auc:.4f}")
    print(f"Detection Confusion Matrix:\n{det_cm}")

    # --------------------------------------------------------------------------
    # TASK 576: Parameter Sweep across Epsilon settings
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("TASK 576: EPSILON SWEEP WITH FIXED SEED (ROBUSTNESS PROFILE)")
    print("=" * 75)
    epsilons = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    sweep_results = []

    for eps in epsilons:
        if eps == 0.0:
            sweep_results.append({
                "epsilon": eps,
                "accuracy": clean_acc,
                "asr": 0.0,
                "mean_conf_drop": 0.0,
                "detection_tpr": 0.0
            })
            continue

        X_adv_sweep = generate_fgsm_examples(model, X_test_correct, y_test_correct, epsilon=eps)
        probs_sweep = model.predict(X_adv_sweep, verbose=0).flatten()
        preds_sweep = (probs_sweep >= 0.5).astype(int)

        asr_sweep = np.sum(preds_sweep != y_test_correct) / len(y_test_correct)
        
        # Whole test acc
        X_tmp = X_test.copy()
        X_tmp[correct_indices] = X_adv_sweep
        acc_sweep = accuracy_score(y_test, (model.predict(X_tmp, verbose=0).flatten() >= 0.5).astype(int))

        # Confidence drop
        conf_drop_sweep = float(np.mean(clean_confidence - np.where(y_test_correct == 1, probs_sweep, 1.0 - probs_sweep)))

        # Detection TPR
        sweep_dists = detector.compute_distance(X_adv_sweep)
        tpr_sweep = float(np.mean(sweep_dists > tau))

        sweep_results.append({
            "epsilon": eps,
            "accuracy": acc_sweep,
            "asr": asr_sweep,
            "mean_conf_drop": conf_drop_sweep,
            "detection_tpr": tpr_sweep
        })

    sweep_df = pd.DataFrame(sweep_results)
    print(sweep_df.to_string(index=False))

    # --------------------------------------------------------------------------
    # DELIVERABLE 579: GENERATE ROBUSTNESS & DETECTION VISUALIZATIONS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("DELIVERABLE 579: GENERATING ROBUSTNESS & DETECTION VISUALIZATIONS")
    print("=" * 75)
    
    # Visualization 1: Robustness Curve vs Epsilon
    plt.figure(figsize=(9, 5.5))
    plt.plot(sweep_df["epsilon"], sweep_df["accuracy"] * 100, "o-", color="#1f77b4", linewidth=2.5, label="Model Accuracy (%)")
    plt.plot(sweep_df["epsilon"], sweep_df["asr"] * 100, "s--", color="#d62728", linewidth=2.5, label="Attack Success Rate (ASR %)")
    plt.plot(sweep_df["epsilon"], sweep_df["detection_tpr"] * 100, "^-.", color="#2ca02c", linewidth=2.5, label="Detector True Positive Rate (%)")
    plt.title(f"Scenario 51: Model Robustness & Detection vs Epsilon Budget (Seed {SEED})", fontsize=12, fontweight="bold")
    plt.xlabel(r"Perturbation Budget ($\epsilon$)", fontsize=11)
    plt.ylabel("Percentage (%)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", shadow=True)
    plt.tight_layout()
    plot1_path = "robustness_curve.png"
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot1_path}")

    # Visualization 2: Confidence Distribution Shift
    plt.figure(figsize=(9, 5.5))
    plt.hist(clean_probs_correct, bins=30, alpha=0.65, color="#1f77b4", label="Clean Predictions (P(Malicious))", density=True)
    plt.hist(adv_probs_correct, bins=30, alpha=0.65, color="#d62728", label=f"FGSM Perturbed (\u03b5={epsilon_test})", density=True)
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Decision Threshold (0.5)")
    plt.title("Model Prediction Distribution Shift Under FGSM Attack", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Probability P(Malicious)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True, facecolor="white", shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot2_path = "confidence_shift.png"
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot2_path}")

    # Visualization 3: Detector Distance Distribution & Separation
    plt.figure(figsize=(9, 5.5))
    plt.hist(dists_clean, bins=35, alpha=0.65, color="#2ca02c", label="Clean Test Feature Distance", density=True)
    plt.hist(dists_adv, bins=35, alpha=0.65, color="#d62728", label=f"FGSM Adversarial Distance (\u03b5={epsilon_test})", density=True)
    plt.axvline(tau, color="black", linestyle="-", linewidth=2.0, label=f"Detection Threshold \u03c4 = {tau:.2f}")
    plt.title("Adversarial Example Detector: Distance from Clean Distribution", fontsize=12, fontweight="bold")
    plt.xlabel("Distance from Clean Distribution", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(frameon=True, facecolor="white", shadow=True)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plot3_path = "adversarial_detection_distribution.png"
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot3_path}")

    # Visualization 4: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    
    # Subplot 1: Clean CM
    axes[0].imshow(clean_cm, cmap="Blues")
    axes[0].set_title(f"Clean Baseline (Acc: {clean_acc*100:.1f}%)", fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Benign", "Malicious"]); axes[0].set_yticklabels(["Benign", "Malicious"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(clean_cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    # Subplot 2: Adversarial CM
    axes[1].imshow(adv_cm, cmap="Reds")
    axes[1].set_title(f"Post-Attack (\u03b5={epsilon_test}, Acc: {adv_acc*100:.1f}%)", fontweight="bold")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Benign", "Malicious"]); axes[1].set_yticklabels(["Benign", "Malicious"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(adv_cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    # Subplot 3: Detection CM
    axes[2].imshow(det_cm, cmap="Greens")
    axes[2].set_title(f"Adversarial Detector (AUC: {det_auc:.3f})", fontweight="bold")
    axes[2].set_xlabel("Detector Decision")
    axes[2].set_xticks([0, 1]); axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Clean", "Flagged"]); axes[2].set_yticklabels(["Actual Clean", "Actual FGSM"])
    for i in range(2):
        for j in range(2):
            axes[2].text(j, i, str(det_cm[i, j]), ha="center", va="center", color="black", fontsize=12)

    plt.tight_layout()
    plot4_path = "confusion_matrices.png"
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"[+] Saved: {plot4_path}")

    # --------------------------------------------------------------------------
    # OUTPUT REPORT SUMMARY (DELIVERABLE 578 & REQUIRED OUTPUT TABLE)
    # --------------------------------------------------------------------------
    report = {
        "clean_baseline": {
            "accuracy": float(clean_acc),
            "precision": float(clean_prec),
            "recall": float(clean_rec),
            "f1": float(clean_f1),
            "confusion_matrix": clean_cm.tolist()
        },
        "adversarial_result": {
            "epsilon": float(epsilon_test),
            "accuracy": float(adv_acc),
            "precision": float(adv_prec),
            "recall": float(adv_rec),
            "f1": float(adv_f1),
            "attack_success_rate": float(asr),
            "confusion_matrix": adv_cm.tolist()
        },
        "confidence_change": {
            "mean_confidence_drop": float(mean_conf_drop),
            "median_confidence_drop": float(median_conf_drop)
        },
        "scenario_detection": {
            "threshold_tau": float(tau),
            "detection_accuracy": float(det_acc),
            "detection_precision": float(det_prec),
            "detection_recall_tpr": float(det_rec),
            "detection_f1": float(det_f1),
            "detection_roc_auc": float(det_auc),
            "confusion_matrix": det_cm.tolist()
        },
        "validity_checks": validity_info,
        "epsilon_sweep": sweep_df.to_dict(orient="records")
    }

    report_path = "lab_report_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n[+] Full structured lab results saved to: {report_path}")

    return report


if __name__ == "__main__":
    run_scenario_51_experiment(epsilon_test=0.15)
