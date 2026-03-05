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

```
Input (5) → Hidden (8) → Hidden (4) → Output (4)
```

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

```python
if   usage < 8%  and power < 20%  → Sleep       (0)
elif usage < 38% and power < 60%  → Low Power   (1)
elif usage < 72% and power < 78%  → Balanced    (2)
else                               → Performance (3)
```

---

## Results

### Float32 Training Results

| Class | Test Samples | Correct | Accuracy |
|---|---|---|---|
| Sleep | 449 | 449 | **100.00%** |
| Low Power | 910 | 831 | **91.32%** |
| Balanced | 834 | 721 | **86.45%** |
| Performance | 807 | 793 | **98.27%** |
| **Overall** | **3,000** | **2,794** | **93.13%** |

### 8-bit Quantization Results

| Metric | Value |
|---|---|
| Float32 Accuracy | 93.13% |
| 8-bit Accuracy | **92.73%** |
| Accuracy Drop | **0.40%** |
| Quantization Method | Dynamic per-layer scaling |

| Class | Float32 | 8-bit | Drop |
|---|---|---|---|
| Sleep | 100.00% | 100.00% | 0.00% |
| Low Power | 91.32% | 92.86% | -1.54% (improved) |
| Balanced | 86.45% | 83.45% | 3.00% |
| Performance | 98.27% | 98.14% | 0.12% |

### Confusion Matrix (Float32)

| | Sleep | Low Power | Balanced | Performance |
|---|---|---|---|---|
| **Sleep** | 449 | 0 | 0 | 0 |
| **Low Power** | 28 | 831 | 51 | 0 |
| **Balanced** | 0 | 76 | 721 | 37 |
| **Performance** | 0 | 0 | 14 | 793 |

---

## Quantization Details

| Layer | Float Min | Float Max | Scale Factor | Int Min | Int Max |
|---|---|---|---|---|---|
| W1 | -1.5574 | 2.1327 | 59.549 | -93 | 127 |
| b1 | -0.2094 | 1.5994 | 79.403 | -17 | 127 |
| W2 | -1.4319 | 3.1184 | 40.726 | -58 | 127 |
| b2 | -0.3285 | 1.2417 | 102.278 | -34 | 127 |
| W3 | -6.7187 | 1.3878 | 18.903 | -127 | 26 |
| b3 | -9.9812 | 7.4932 | 12.724 | -127 | 95 |

---

## How to Run

### 1. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib torch torchvision
```

### 2. Collect HWiNFO64 data
- Run HWiNFO64 on your machine
- Log 5 sessions: idle, light, medium, heavy, burst
- Place CSV files in the project folder

### 3. Run preprocessing
```bash
python preprocess.py
```

### 4. Train the model
```bash
python train.py
```

### 5. Quantize weights for Verilog
```bash
python quantize.py
```

---

## Project Structure

```
nn-power-management/
├── preprocess.py
├── train.py
├── quantize.py
├── README.md
│
├── dataset_output/
│   ├── feature_stats.csv
│   ├── class_weights.csv
│   ├── feature_distributions.png
│   ├── usage_vs_slack_scatter.png
│   └── raw_cpu_usage_per_session.png
│
├── quantization_output/
│   ├── weights_layer1.mem
│   ├── weights_layer2.mem
│   ├── weights_layer3.mem
│   ├── layer_scales.csv
│   ├── weights_quantized.csv
│   ├── weights_summary.txt
│   └── quantization_analysis.png
│
└── training_output/
    ├── confusion_matrix.csv
    ├── confusion_matrix.png
    └── training_curves.png
```

---

## Feature Normalization Ranges (for FPGA deployment)

| Feature | Min | Max |
|---|---|---|
| Frequency (MHz) | 115.30 | 3955.10 |
| Temperature (°C) | 50.00 | 94.10 |
| CPU Usage (%) | 1.20 | 100.00 |
| Package Power (W) | 4.10 | 39.66 |
| Voltage (V) | 0.76 | 1.46 |

---

## Roadmap

- [x] Phase 1 — Data Collection (HWiNFO64, 5 sessions, 6,326 samples)
- [x] Phase 2 — Preprocessing (normalization, labeling, augmentation to 20,000)
- [x] Phase 3 — MLP Training (93.13% test accuracy)
- [x] Phase 4 — 8-bit Quantization (92.73%, only 0.40% drop)
- [ ] Phase 5 — RTL implementation in Verilog
- [ ] Phase 6 — FPGA synthesis and verification
