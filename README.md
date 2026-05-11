# Neural Network-Based Adaptive Power Management for VLSI Systems

**Institution:** BMS College of Engineering, Bangalore  
**Department:** Electronics and Communication Engineering  

---

## Overview

A lightweight MLP (Multi-Layer Perceptron) classifier that predicts the optimal CPU power mode in real time based on hardware telemetry. The model is trained on real sensor data collected from an ASUS Vivobook (AMD Ryzen 7 5825U) using HWiNFO64, and implemented as a synthesizable RTL FSM in Verilog, verified through behavioral simulation in Vivado 2025.2, synthesized on Artix-7 FPGA, and implemented as a full custom ASIC in Cadence Innovus using 32nm SAED technology.

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
  (20000 samples)
        ↓
  FPGA Synthesis
  (Artix-7 xc7a12ticsg325-1L)
        ↓
  ASIC Implementation
  (Cadence Innovus, 32nm SAED)
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

### Confusion Matrix (Float32)

| | Sleep | Low Power | Balanced | Performance |
|---|---|---|---|---|
| **Sleep** | 453 | 0 | 0 | 0 |
| **Low Power** | 37 | 797 | 61 | 0 |
| **Balanced** | 0 | 71 | 734 | 32 |
| **Performance** | 0 | 0 | 12 | 803 |

### Complete Accuracy Comparison

| Test Set | Samples | Overall | Sleep | LowPower | Balanced | Performance |
|---|---|---|---|---|---|---|
| Float32 Python | 3,000 | 92.90% | 100% | 89% | 88% | 99% |
| Fixed-Point Python | 6,326 | 93.50% | 100% | 87% | 91% | 99% |
| Verilog XSim | 6,326 | 93.50% | 100% | 87% | 91% | 99% |

> Python fixed-point and Verilog simulation produce **identical results** on every test case — confirming the hardware is a bit-accurate implementation of the software model.

### Baseline Comparison — MLP vs Windows DVFS

| Method | Accuracy | Notes |
|---|---|---|
| Windows Balanced AC (80% floor) | ~65% | powercfg PROCTHROTTLEMIN=0x50 |
| Linux schedutil governor | 71.8% | Best software governor |
| **MLP Fixed-Point (hardware)** | **93.5%** | **+21.7pp over schedutil** |

> Windows AC mode enforces 80% minimum CPU frequency floor, preventing Sleep/LowPower states entirely. MLP correctly prescribes Sleep (4.5W) and LowPower (8.0W) for 44.9% of workloads that Windows misses.

---

## Fixed-Point Quantization

Floating-point weights are scaled by 1000× and stored as 16-bit signed integers. Biases are scaled by 1,000,000 (1000 × 1000). Between layers, accumulators are right-shifted by 10 bits (÷1024 ≈ ÷1000) to restore magnitude before the next layer. All intermediate values use 48-bit signed accumulators to prevent overflow.

This avoids any floating-point arithmetic in hardware while preserving classification accuracy.

---

## Hardware Implementation

The design is split into 4 Verilog modules inside `power_mlp_top.v`:

| Module | Function |
|---|---|
| `layer1` | W1×x + B1, ReLU, >>>10 → h1[0:7] |
| `layer2` | W2×h1 + B2, ReLU, >>>10 → h2[0:3] |
| `layer3` | W3×h2 + B3, no ReLU → z3[0:3] |
| `power_mlp_top` | Sequencer FSM + argmax → mode[1:0] |

**Design parameters:**
- Inputs: 5 × 11-bit unsigned (range 0–1000)
- Weights: 16-bit signed integers (hardcoded)
- Accumulators: 48-bit signed
- ReLU: inline, clips negative accumulator to zero then >>>10
- Output: 2-bit mode signal + valid pulse
- Latency: 127 clock cycles @ 100 MHz = **1.27 µs per inference**
- No floating-point units required

**Top-level FSM States:**
```
IDLE → S_L1 → S_L2 → S_L3 → S_OUT → IDLE
```

---

## FPGA Resource Utilization

Target device: **Xilinx Artix-7 xc7a12ticsg325-1L**

| Resource | Used | Available | Utilization |
|---|---|---|---|
| Slice LUTs | 834 | 8,000 | **10.43%** |
| Flip Flops | 829 | 16,000 | **5.18%** |
| DSP48E1 | 7 | 40 | **17.50%** |
| Block RAM | 0 | 20 | **0.00%** |
| I/O Pins | 61 | 150 | **40.67%** |
| Clock Buffer | 1 | 32 | **3.13%** |

---

## ASIC Implementation — Cadence Innovus (32nm SAED)

| Metric | Value |
|---|---|
| Technology | 32nm SAED LVT |
| Standard Cells | 6,068 |
| Total Area | 23,654 µm² |
| Estimated Power | 0.88 mW |
| Clock Frequency | 100 MHz |
| Setup Slack | +5.999 ns (timing met) |
| Equivalent Frequency | ~200 MHz achievable |

> Power estimated using default 20% toggle rate. VCD-based analysis pending.

### Implementation vs FPGA Comparison

| Metric | FPGA (Artix-7) | ASIC (32nm SAED) |
|---|---|---|
| Area | 834 LUTs | 23,654 µm² |
| Power | ~12.4 W (board) | 0.88 mW |
| Clock | 100 MHz | 100 MHz (+6ns slack) |
| Latency | 1.27 µs | 1.27 µs |
| Power Reduction | — | **~14,000×** |

---

## Simulation Results — 6326 Real Samples

```
=====================================================
   Power MLP FSM - 6326 Real CSV Samples
=====================================================
  Sleep       : 967/967   (100%)
  LowPower    : 1648/1885  (87%)
  Balanced    : 1624/1774  (91%)
  Performance : 1683/1700  (99%)
-----------------------------------------------------
  Overall     : 5922/6326  (93%)
=====================================================
```

---

## Feature Normalization Ranges (for FPGA/ASIC deployment)

| Feature | Min | Max |
|---|---|---|
| Frequency (MHz) | 115.30 | 3955.10 |
| Temperature (°C) | 50.00 | 94.10 |
| CPU Usage (%) | 1.20 | 100.00 |
| Package Power (W) | 4.10 | 39.66 |

---

## Power Estimation Model (Ryzen 7 5825U)

| Mode | Estimated TDP |
|---|---|
| Sleep | 4.5 W |
| Low Power | 8.0 W |
| Balanced | 15.0 W |
| Performance | 25.0 W |

> MLP prescribes Sleep/LowPower for 44.9% of workloads — Windows AC mode cannot drop below Balanced (15W minimum due to 80% frequency floor).

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

### 5. Quantize and generate test vectors
```bash
python quantize.py
python test.py
```

### 6. Run Verilog simulation
- Open `power_mlp_fsm` project in Vivado 2025.2
- Copy desired `.txt` file from `fixedpoint_output/` to the xsim working directory
- Run Behavioral Simulation
- In Tcl Console: `run 15000ms`

---

## Repository Structure

```
nn-power-management/
├── preprocess.py               # Data cleaning, augmentation, scaling
├── train.py                    # MLP training (PyTorch)
├── quantize.py                 # Fixed-point quantization analysis
├── test.py                     # Fixed-point weight generation
├── new_vectors.py              # Dataset generation for Verilog
├── baseline_comparison.py      # MLP vs DVFS governors comparison
├── verilog/
│   ├── power_mlp_top.v         # RTL implementation (4 modules)
│   └── tb_power_mlp.v          # Testbench
├── fixedpoint_output/
│   ├── testvectors.txt          # 20 verified vectors (set 1)
│   ├── testvectors2.txt         # 20 verified vectors (set 2)
│   ├── testvectors1000.txt      # 1000 test vectors
│   └── testvectors_real.txt     # 6326 real CSV vectors
├── dataset_output/
│   ├── feature_stats.csv        # Normalization ranges
│   └── class_weights.csv        # Class weights for training
└── quantization_output/
    └── weights_summary.txt      # Quantized weight values
```

---

## Tools Used

| Tool | Purpose |
|---|---|
| HWiNFO64 | Real CPU telemetry data collection |
| Python 3 + PyTorch | Data preprocessing and model training |
| NumPy | Fixed-point arithmetic verification |
| Vivado 2025.2 | RTL synthesis and simulation |
| Cadence Innovus | Physical design — place and route |
| SAED 32nm PDK | Standard cell library for ASIC |

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
- [x] Phase 4 — Fixed-Point Quantization (1000× scale, 48-bit accumulators)
- [x] Phase 5 — RTL Implementation in Verilog (4 modules, hierarchical FSM)
- [x] Phase 6 — Behavioral Simulation (6,326 real samples, 93.5% accuracy)
- [x] Phase 7 — FPGA Synthesis (834 LUTs, 7 DSPs, 10% utilization on Artix-7)
- [x] Phase 8 — ASIC Physical Design (Cadence Innovus, 32nm, 0.88mW, 23,654µm²)
