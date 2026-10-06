# Adversarial Example Detection in Cybersecurity 🛡️⚡

An end-to-end framework for investigating, generating, and detecting **Adversarial Examples** against Deep Learning-based Network Intrusion Detection Systems (NIDS) using the **Fast Gradient Sign Method (FGSM)**.

---

## 📌 Project Overview

Deep learning models deployed in cybersecurity operations (e.g., Network Intrusion Detection Systems) are susceptible to adversarial evasion attacks. Small, carefully crafted perturbations added to network traffic telemetry can mislead classifiers into misclassifying malicious attacks as benign traffic.

This repository implements:
1. **Benchmark Cybersecurity Dataset Generation & Preprocessing**: Realistic network flow features (duration, byte counts, error rates, host rates, protocols, services) simulating benign traffic and intrusions (DoS Syn-Flood, PortScan, Brute Force).
2. **Deep Neural Network Intrusion Classifier**: Multi-layer perceptron architecture with batch normalization, dropout, and bounded activations.
3. **FGSM Adversarial Perturbation Engine**: Generates $\ell_\infty$-bounded adversarial examples to assess model vulnerability.
4. **Adversarial Detection & Defense**: Distance/confidence-based anomaly detection mechanism to filter and flag adversarial traffic.
5. **Robustness Evaluation & Analytics**: Epsilon sweep experiments ($\epsilon \in [0.0, 0.25]$), evaluating Attack Success Rate (ASR), confidence degradation, detection True Positive Rate (TPR), ROC-AUC, and confusion matrices.

---

## 📊 Visual Results

| Metric / Analysis | Visualization |
| :--- | :--- |
| **Model Robustness & Epsilon Sweep** | ![Robustness Curve](robustness_curve.png) |
| **Detection Score Distribution** | ![Adversarial Detection Distribution](adversarial_detection_distribution.png) |
| **Confidence Degradation** | ![Confidence Shift](confidence_shift.png) |
| **Confusion Matrices (Clean vs Adv vs Detection)** | ![Confusion Matrices](confusion_matrices.png) |

---

## 📂 Repository Structure

```plaintext
├── prepare_dataset.py                    # Dataset generation, one-hot encoding & MinMax scaling
├── adversarial_fgsm_detector.py          # PyTorch/TensorFlow classifier, FGSM attack & detection engine
├── requirements.txt                      # Python dependencies
├── preprocessing_metadata.json           # Normalization statistics and feature bounds
├── lab_report_results.json               # Quantitative benchmark metrics and evaluation results
├── cybersecurity_features_raw.csv        # Raw synthesized network flow features
├── cybersecurity_train_clean.csv         # Clean normalized training dataset
├── cybersecurity_test_clean.csv          # Clean normalized testing dataset
├── robustness_curve.png                  # Epsilon perturbation sweep plot
├── adversarial_detection_distribution.png# Anomaly detection score histogram
├── confidence_shift.png                  # Model prediction confidence shift under attack
└── confusion_matrices.png                # Multi-stage confusion matrices
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/srihariharancyber/Adversarial-Example-Detection.git
   cd Adversarial-Example-Detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 1. Generate & Preprocess Dataset
```bash
python prepare_dataset.py
```
Outputs:
- `cybersecurity_features_raw.csv`
- `cybersecurity_train_clean.csv`
- `cybersecurity_test_clean.csv`
- `preprocessing_metadata.json`

### 2. Train Model & Run Adversarial Detection
```bash
python adversarial_fgsm_detector.py
```
This script will:
- Train the neural network on clean cybersecurity flow samples.
- Craft $\ell_\infty$-bounded adversarial examples using FGSM.
- Evaluate adversarial evasion capability and detection TPR/FPR.
- Generate all analytical figures (`.png`) and report `lab_report_results.json`.

---

## 📈 Key Findings & Results

- **Baseline Clean Accuracy**: 100% on test partition.
- **Validity & Bounds**: All perturbations strictly obey $\ell_\infty \le \epsilon$ and clip to valid feature bounds $[0, 1]$.
- **Detector Performance**: The statistical detection mechanism flags adversarial anomalies with strong ROC-AUC discrimination prior to classification hand-off.

---

## 📜 License

MIT License. Free for academic and research use.
