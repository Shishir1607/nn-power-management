# Neural Network Based Adaptive Power Management

**Institution:** BMS College of Engineering, Bangalore  
**Department:** Electronics and Communication Engineering  
**Project Type:** VLSI Systems — Mini Project  

---

## Overview

A lightweight MLP (Multi-Layer Perceptron) classifier that predicts the optimal CPU power mode in real time based on hardware telemetry. The model is trained on real sensor data collected from an ASUS Vivobook (AMD Ryzen 7 5825U) using HWiNFO64, and is designed for eventual deployment as an RTL module on FPGA.

The system approximates **DVFS (Dynamic Voltage and Frequency Scaling)** behaviour — predicting which power mode the CPU should operate in based on current workload and thermal state. The actual voltage and frequency scaling is handled by the hardware after receiving the mode decision.

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
Input (5) → Hidden Layer 1 (8) → Hidden Layer 2 (4) → Output (4)
```

| Property | Value |
|---|---|
| Architecture | 5 → 8 → 4 → 4 MLP |
| Total Parameters | 104 |
| Activation | ReLU |
| Loss Function | CrossEntropyLoss (weighted) |
| Optimizer | Adam (lr=0.001) |
| Best Epoch | 52 |

---

## Input Features

| # | Feature | Source | Description |
|---|---|---|---|
| 1 | freq_norm | Average Effective Clock [MHz] | Normalized CPU frequency |
| 2 | temp_norm | CPU Tctl/Tdie [°C] | Normalized die temperature |
| 3 | usage_norm | Total CPU Usage [%] | Normalized CPU utilization |
| 4 | switching_activity | Derived from Package Power [W] | Approximates transistor toggle rate |
| 5 | timing_slack | Derived from usage + temp | Headroom before thermal/perf limit |

> **Note:** Voltage was intentionally excluded. It is highly correlated with frequency and carries redundant information. Voltage is an output effect of the power mode decision, not an input to it.

---

## Derived Features

```
switching_activity = power_norm
timing_slack       = (1 − usage_norm) × (1 − temp_norm)
```

**Timing Slack Examples:**

| Condition | Calculation | Slack | Meaning |
|---|---|---|---|
| Idle (usage=5%, temp=62°C) | (1−0.053)×(1−0.272) | **0.689** | Lots of headroom |
| Heavy (usage=97%, temp=76°C) | (1−0.969)×(1−0.594) | **0.013** | Almost no headroom |

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

| Session | Rows | CPU Mean | Temp Mean | Power Mean | Description |
|---|---|---|---|---|---|
| Idle | 1,188 | 6.4% | 62.3°C | 8.3W | No applications running |
| Light | 1,215 | 20.1% | 92.4°C | 26.1W | Google Meet call |
| Medium | 1,077 | 39.2% | 85.8°C | 23.0W | Meet + Chrome tabs + VS Code |
| Heavy | 1,204 | 96.9% | 76.4°C | 23.4W | Prime95 stress test |
| Burst | 1,642 | 33.0% | 88.0°C | 26.9W | Alternating load cycles |

### Class Distribution

| Label | Class | Raw Count | % |
|---|---|---|---|
| 0 | Sleep | 967 | 15% |
| 1 | Low Power | 1,885 | 30% |
| 2 | Balanced | 1,774 | 28% |
| 3 | Performance | 1,700 | 27% |

### Labeling Thresholds

```python
if   usage_norm < 0.08 and switching_activity < 0.20  →  Sleep       (0)
elif usage_norm < 0.38 and switching_activity < 0.60  →  Low Power   (1)
elif usage_norm < 0.72 and switching_activity < 0.78  →  Balanced    (2)
else                                                   →  Performance (3)
```

---

## Results

### Float32 Training Results

| Class | Test Samples | Correct | Accuracy |
|---|---|---|---|
| Sleep | 453 | 453 | **100.00%** |
| Low Power | 895 | 797 | **89.05%** |
| Balanced | 837 | 734 | **87.69%** |
| Performance | 815 | 803 | **98.53%** |
| **Overall** | **3,000** | **2,787** | **92.90%** |

### 8-bit Quantization Results

| Metric | Value |
|---|---|
| Float32 Accuracy | 92.90% |
| 8-bit Accuracy | **92.77%** |
| Accuracy Drop | **0.13%** |
| Quantization Method | Dynamic per-layer scaling |

### Confusion Matrix (Float32)

| | Sleep | Low Power | Balanced | Performance |
|---|---|---|---|---|
| **Sleep** | 453 | 0 | 0 | 0 |
| **Low Power** | 37 | 797 | 61 | 0 |
| **Balanced** | 0 | 71 | 734 | 32 |
| **Performance** | 0 | 0 | 12 | 803 |

---

## Quantization Details

Scale factors saved in `quantization_output/layer_scales.csv`.  
Quantized weights saved as `.mem` files for Verilog `$readmemh`.

---

## Feature Normalization Ranges (for FPGA deployment)

| Feature | Min | Max |
|---|---|---|
| Frequency (MHz) | 115.30 | 3955.10 |
| Temperature (°C) | 50.00 | 94.10 |
| CPU Usage (%) | 1.20 | 100.00 |
| Package Power (W) | 4.10 | 39.66 |

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
│   ├── feature_stats.csv               — Min/max for FPGA normalization
│   ├── class_weights.csv               — Per-class loss weights
│   ├── feature_distributions.png       — Feature histograms per class
│   ├── usage_vs_slack_scatter.png      — Label boundary visualization
│   └── raw_cpu_usage_per_session.png   — Raw session comparison
│
├── quantization_output/
│   ├── weights_layer1.mem              — Layer 1 weights for Verilog
│   ├── weights_layer2.mem              — Layer 2 weights for Verilog
│   ├── weights_layer3.mem              — Layer 3 weights for Verilog
│   ├── layer_scales.csv                — Per-layer scale factors
│   ├── weights_quantized.csv           — All 104 weights as integers
│   ├── weights_summary.txt             — Human readable values
│   └── quantization_analysis.png       — Quantization error plots
│
└── training_output/
    ├── confusion_matrix.csv            — Confusion matrix data
    ├── confusion_matrix.png            — Confusion matrix heatmap
    └── training_curves.png             — Loss and accuracy curves
```

---

## Roadmap

- [x] Phase 1 — Data Collection (HWiNFO64, 5 sessions, 6,326 real samples)
- [x] Phase 2 — Preprocessing (normalization, labeling, augmentation to 20,000)
- [x] Phase 3 — MLP Training (92.90% test accuracy, converged at epoch 52)
- [x] Phase 4 — 8-bit Quantization (92.77%, only 0.13% accuracy drop)
- [ ] Phase 5 — RTL implementation in Verilog
- [ ] Phase 6 — FPGA synthesis and verification
