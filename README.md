# Neural Network-Based Adaptive Power Management for VLSI Systems

**Institution:** BMS College of Engineering, Bangalore  
**Department:** Electronics and Communication Engineering  
**Project Type:** VLSI Systems — Project Work 1

---

## Overview

A lightweight MLP (Multi-Layer Perceptron) classifier that predicts the optimal CPU power mode in real time based on hardware telemetry. The model is trained on real sensor data collected from an ASUS Vivobook (AMD Ryzen 7 5825U) using HWiNFO64, and implemented as a synthesizable RTL FSM in Verilog, verified through behavioral simulation in Vivado 2025.2.

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

## System Architecture

```
Real CPU Data (HWiNFO64)
        ↓
  preprocess.py
  (augment, scale, split)
        ↓
    train.py
  (MLP 5→8→4→4, ReLU)
        ↓
   quantize.py
  (1000x fixed-point)
        ↓
 power_mlp_top.v
  (FSM in Verilog)
        ↓
  Vivado Simulation
  (20/20 test vectors)
```

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

### Fixed-Point Quantization Results

| Metric | Value |
|---|---|
| Float32 Accuracy | 92.90% |
| 1000x Fixed-Point Accuracy | **92.43%** |
| Accuracy Drop | **0.47%** |
| Quantization Method | 1000x weight scaling, 48-bit accumulators |

### Confusion Matrix (Float32)

| | Sleep | Low Power | Balanced | Performance |
|---|---|---|---|---|
| **Sleep** | 453 | 0 | 0 | 0 |
| **Low Power** | 37 | 797 | 61 | 0 |
| **Balanced** | 0 | 71 | 734 | 32 |
| **Performance** | 0 | 0 | 12 | 803 |

---

## Fixed-Point Quantization

Floating-point weights are scaled by 1000x and stored as 16-bit signed integers. Biases are scaled by 1,000,000 (1000 × 1000). Between layers, accumulators are right-shifted by 10 bits (÷1024 ≈ ÷1000) to restore magnitude before the next layer. All intermediate values use 48-bit signed accumulators to prevent overflow.

This avoids any floating-point arithmetic in hardware while preserving classification accuracy.

---

## Hardware Implementation

**File:** `verilog/power_mlp_top.v`

- Clocked synchronous FSM with 10 states
- Inputs: 5 × 11-bit unsigned (range 0–1000)
- Weights: 16-bit signed integers (hardcoded)
- Accumulators: 48-bit signed
- ReLU: inline, clips negative accumulator to zero
- Output: 2-bit mode signal + valid pulse
- No floating-point units required

**FSM States:**
```
IDLE → L1_MAC → L1_BIAS → L1_RELU
     → L2_MAC → L2_BIAS → L2_RELU
     → L3_MAC → L3_BIAS → OUTPUT → IDLE
```

---

## Simulation Results

Verified in Vivado 2025.2 behavioral simulation using 20 test vectors (5 per class):

```
=====================================================
   Power MLP FSM - Behavioral Simulation
=====================================================
 # |  f0   f1   f2   f3   f4 | Exp | Got | Result
---+---------------------------+-----+-----+-------
 1 |  133  880  111  506  107 | 1 | 1 | PASS
 2 |  203  995  185  622    4 | 2 | 2 | PASS
 3 |  837  567  993  532    3 | 3 | 3 | PASS
 4 |   83  946   98  438   53 | 1 | 1 | PASS
 5 |  836  555 1000  557    0 | 3 | 3 | PASS
 6 |  285  780  308  527  152 | 1 | 1 | PASS
 7 |  822  596 1000  532    0 | 3 | 3 | PASS
 8 |   43   75   48   75  945 | 0 | 0 | PASS
 9 |  840  585  994  532    3 | 3 | 3 | PASS
10 |  859  895  912  780    0 | 3 | 3 | PASS
11 |  265  941  269  655   43 | 2 | 2 | PASS
12 |  263  961  249  624   29 | 2 | 2 | PASS
13 |  124  723  118  512  244 | 1 | 1 | PASS
14 |  288  859  286  523  108 | 1 | 1 | PASS
15 |  547  725  573  530  134 | 2 | 2 | PASS
16 |   13    6   20   29  942 | 0 | 0 | PASS
17 |  173 1000  236  629   37 | 2 | 2 | PASS
18 |    1   11   11   34  977 | 0 | 0 | PASS
19 |   37  160   18   50  824 | 0 | 0 | PASS
20 |   10  163   35   47  807 | 0 | 0 | PASS
=====================================================
  Total    : 20
  Pass     : 20
  Fail     : 0
  Accuracy : 100%
=====================================================
```

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

### 5. Generate fixed-point weights and test vectors
```bash
python quantize.py
python test.py
```

### 6. Run Verilog simulation
- Open `power_mlp_fsm` project in Vivado 2025.2
- Copy `fixedpoint_output/testvectors.txt` to the xsim working directory
- Run Behavioral Simulation
- In Tcl Console: `run 5ms`

---

## Repository Structure

```
nn-power-management/
├── preprocess.py               # Data cleaning, augmentation, scaling
├── train.py                    # MLP training (PyTorch)
├── quantize.py                 # 8-bit quantization analysis
├── test.py                     # Fixed-point weight generation and test vectors
├── verilog/
│   ├── power_mlp_top.v         # RTL implementation (FSM)
│   └── tb_power_mlp.v          # Testbench with 20 test vectors
└── fixedpoint_output/
    └── testvectors.txt         # 20 verified test vectors
```

---

## Tools Used

| Tool | Purpose |
|------|---------|
| HWiNFO64 | Real CPU telemetry data collection |
| Python 3 + PyTorch | Data preprocessing and model training |
| NumPy | Fixed-point arithmetic verification |
| Vivado 2025.2 | RTL simulation and synthesis |

---

## Hardware Platform

- **CPU:** AMD Ryzen 7 5825U
- **Device:** ASUS Vivobook M1502YA
- **OS:** Windows 11

---

## Roadmap

- [x] Phase 1 — Data Collection (HWiNFO64, 5 sessions, 6,326 real samples)
- [x] Phase 2 — Preprocessing (normalization, labeling, augmentation to 20,000)
- [x] Phase 3 — MLP Training (92.90% test accuracy, converged at epoch 52)
- [x] Phase 4 — Fixed-Point Quantization (92.43%, only 0.47% accuracy drop)
- [x] Phase 5 — RTL Implementation in Verilog (FSM, 10 states)
- [x] Phase 6 — Behavioral Simulation (20/20 test vectors, 100% accuracy)
- [ ] Phase 7 — FPGA Synthesis and On-Device Verification

---
