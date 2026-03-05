# Neural Network Based Adaptive Power Management

**Institution:** BMS College of Engineering, Bangalore  
**Department:** Electronics and Communication Engineering  
**Project Type:** VLSI Systems — Mini Project  

---

## Overview

A lightweight MLP (Multi-Layer Perceptron) classifier that predicts the optimal CPU power mode in real time based on hardware telemetry. The model is trained on real sensor data collected from an ASUS Vivobook (AMD Ryzen 7 5825U) using HWiNFO64, and is designed for eventual deployment as an RTL module on FPGA.

---

## Power Modes

| Label | Mode | Description |
|---|---|---|
| 0 | Sleep | Very low activity, maximum power saving |
| 1 | Low Power | Light workload, efficient operation |
| 2 | Balanced | Moderate workload, balanced performance |
| 3 | Performance | Heavy workload, maximum performance |

---

## Model Architecture


Input (5) → Hidden (8) → Hidden (4) → Output (4)


| Property | Value |
|---|---|
| Architecture | 5 → 8 → 4 → 4 MLP |
| Total Parameters | 104 |
| Activation | ReLU |
| Loss Function | CrossEntropyLoss (weighted) |
| Optimizer | Adam (lr=0.001) |
| Best Epoch | 82 |

---

## Input Features

| # | Feature | Source | Description |
|---|---|---|---|
| 1 | CPU Usage (normalized) | Total CPU Usage [%] | Direct sensor reading |
| 2 | Temperature (normalized) | CPU Tctl/Tdie [°C] | Direct sensor reading |
| 3 | Package Power (normalized) | CPU Package Power [W] | Direct sensor reading |
| 4 | Switching Activity | Derived from power | Approximates transistor toggle rate |
| 5 | Timing Slack | Derived from usage + temp | Headroom before thermal/perf limit |

---

## Dataset

| Property | Value |
|---|---|
| Hardware | ASUS Vivobook M1502YA — AMD Ryzen 7 5825U |
| Collection Tool | HWiNFO64 |
| Real Samples | 6,326 rows |
| Augmented Total | 20,000 samples |
| Train / Val / Test | 14,000 / 3,000 / 3,000 |

### Workload Sessions

| Session | Rows | CPU Mean | Description |
|---|---|---|---|
| Idle | 1,188 | 6.4% | No applications running |
| Light | 1,215 | 20.1% | Google Meet call |
| Medium | 1,077 | 39.2% | Meet + Chrome tabs + VS Code |
| Heavy | 1,204 | 96.9% | Prime95 stress test |
| Burst | 1,642 | 33.0% | Alternating load cycles |

### Labeling Thresholds


- if   usage < 8%  and power < 20%  → Sleep       (0)
- elif usage < 38% and power < 60%  → Low Power   (1)
- elif usage < 72% and power < 78%  → Balanced    (2)
- else                              → Performance (3)


---

## Results

| Class | Test Samples | Correct | Accuracy |
|---|---|---|---|
| Sleep | 449 | 449 | **100.00%** |
| Low Power | 910 | 831 | **91.32%** |
| Balanced | 834 | 721 | **86.45%** |
| Performance | 807 | 793 | **98.27%** |
| **Overall** | **3,000** | **2,794** | **93.13%** |

### Confusion Matrix

| | Sleep | Low Power | Balanced | Performance |
|---|---|---|---|---|
| **Sleep** | 449 | 0 | 0 | 0 |
| **Low Power** | 28 | 831 | 51 | 0 |
| **Balanced** | 0 | 76 | 721 | 37 |
| **Performance** | 0 | 0 | 14 | 793 |

---

## Project Structure
```
nn-power-management/
│
├── preprocess.py
├── train.py
├── README.md
│
├── dataset_output/
│   ├── feature_stats.csv
│   ├── class_weights.csv
│   ├── feature_distributions.png
│   ├── usage_vs_slack_scatter.png
│   └── raw_cpu_usage_per_session.png
│
└── training_output/
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    └── training_curves.png
```

---

## How to Run

### 1. Install dependencies

pip install pandas numpy scikit-learn matplotlib torch torchvision


### 2. Collect HWiNFO64 data
- Run HWiNFO64 on your machine
- Log 5 sessions: idle, light, medium, heavy, burst
- Place CSV files in the project folder

### 3. Run preprocessing

python preprocess.py


### 4. Train the model

python train.py


---

## Roadmap

- [x] Phase 1 — Data Collection (HWiNFO64, 5 sessions)
- [x] Phase 2 — Preprocessing (normalization, labeling, augmentation)
- [x] Phase 3 — MLP Training (93.13% test accuracy)
- [ ] Phase 4 — Quantization and weight extraction
- [ ] Phase 5 — RTL implementation in Verilog
- [ ] Phase 6 — FPGA synthesis and verification

---

## Feature Normalisation Ranges (for FPGA deployment)

| Feature | Min | Max |
|---|---|---|
| Frequency (MHz) | 115.30 | 3955.10 |
| Temperature (°C) | 50.00 | 94.10 |
| CPU Usage (%) | 1.20 | 100.00 |
| Package Power (W) | 4.10 | 39.66 |
| Voltage (V) | 0.76 | 1.46 |


