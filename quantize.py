# =============================================================
# Neural Network Based Adaptive Power Management
# Phase 4: Quantization & Weight Extraction
# 8-bit dynamic per-layer quantization for Verilog RTL
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# =============================================================
# CONFIGURATION
# =============================================================

DATA_DIR   = 'dataset_output'
TRAIN_DIR  = 'training_output'
OUTPUT_DIR = 'quantization_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

BIT_WIDTH  = 8
label_names = {0: 'Sleep', 1: 'Low Power', 2: 'Balanced', 3: 'Performance'}

print("=" * 60)
print("Phase 4 — Quantization & Weight Extraction")
print(f"Bit width: {BIT_WIDTH}-bit | Dynamic per-layer scaling")
print("=" * 60)

# =============================================================
# STEP 1 — LOAD MODEL
# =============================================================

print("\n[1/5] Loading trained model...")
print("-" * 40)

class PowerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 4)
        )
    def forward(self, x):
        return self.net(x)

model = PowerMLP()
model.load_state_dict(torch.load(os.path.join(TRAIN_DIR, 'best_model.pth')))
model.eval()

W1 = model.net[0].weight.data.numpy()
b1 = model.net[0].bias.data.numpy()
W2 = model.net[2].weight.data.numpy()
b2 = model.net[2].bias.data.numpy()
W3 = model.net[4].weight.data.numpy()
b3 = model.net[4].bias.data.numpy()

print(f"  Model loaded | Total params: {W1.size+b1.size+W2.size+b2.size+W3.size+b3.size}")

# =============================================================
# STEP 2 — QUANTIZE (dynamic per-layer scaling)
# =============================================================

print("\n[2/5] Quantizing weights to 8-bit...")
print("-" * 40)

def quantize_layer(arr, bits=8):
    max_val   = np.abs(arr).max()
    scale     = (2**(bits-1) - 1) / max_val
    quantized = np.round(arr * scale).astype(np.int32)
    quantized = np.clip(quantized, -(2**(bits-1)), (2**(bits-1))-1)
    return quantized, scale

W1_q, s_W1 = quantize_layer(W1)
b1_q, s_b1 = quantize_layer(b1)
W2_q, s_W2 = quantize_layer(W2)
b2_q, s_b2 = quantize_layer(b2)
W3_q, s_W3 = quantize_layer(W3)
b3_q, s_b3 = quantize_layer(b3)

print(f"  {'Layer':>8} | {'Float Min':>10} | {'Float Max':>10} | {'Scale':>8} | {'Int Min':>8} | {'Int Max':>8}")
print(f"  {'-'*68}")
for name, orig, quant, scale in [
    ('W1', W1, W1_q, s_W1), ('b1', b1, b1_q, s_b1),
    ('W2', W2, W2_q, s_W2), ('b2', b2, b2_q, s_b2),
    ('W3', W3, W3_q, s_W3), ('b3', b3, b3_q, s_b3),
]:
    print(f"  {name:>8} | {orig.min():>10.4f} | {orig.max():>10.4f} | "
          f"{scale:>8.3f} | {quant.min():>8d} | {quant.max():>8d}")

scales = {'W1':s_W1,'b1':s_b1,'W2':s_W2,'b2':s_b2,'W3':s_W3,'b3':s_b3}
pd.DataFrame([{'parameter':k,'scale':v} for k,v in scales.items()]).to_csv(
    os.path.join(OUTPUT_DIR, 'layer_scales.csv'), index=False)
print(f"\n  Layer scale factors saved to layer_scales.csv")

# =============================================================
# STEP 3 — VERIFY ACCURACY
# =============================================================

print("\n[3/5] Verifying accuracy with quantized weights...")
print("-" * 40)

X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

def relu(x):
    return np.maximum(0, x)

def forward_float(X):
    h1  = relu(X @ W1.T + b1)
    h2  = relu(h1 @ W2.T + b2)
    return np.argmax(h2 @ W3.T + b3, axis=1)

def forward_quantized(X):
    W1_f = W1_q/s_W1; b1_f = b1_q/s_b1
    W2_f = W2_q/s_W2; b2_f = b2_q/s_b2
    W3_f = W3_q/s_W3; b3_f = b3_q/s_b3
    h1  = relu(X @ W1_f.T + b1_f)
    h2  = relu(h1 @ W2_f.T + b2_f)
    return np.argmax(h2 @ W3_f.T + b3_f, axis=1)

preds_float = forward_float(X_test)
preds_quant = forward_quantized(X_test)
acc_float   = (preds_float == y_test).mean() * 100
acc_quant   = (preds_quant == y_test).mean() * 100
acc_drop    = acc_float - acc_quant

print(f"  Float32 accuracy  : {acc_float:.2f}%")
print(f"  8-bit   accuracy  : {acc_quant:.2f}%")
print(f"  Accuracy drop     : {acc_drop:.2f}%")

if acc_drop < 2.0:
    print(f"  ✓ Excellent — drop within 2%")
elif acc_drop < 5.0:
    print(f"  ✓ Acceptable — drop within 5%")
else:
    print(f"  ✗ Drop too large — may need 16-bit")

print(f"\n  Per-class accuracy:")
print(f"  {'Class':>12} | {'Float':>7} | {'Quantized':>9} | {'Drop':>6}")
print(f"  {'-'*45}")
for lid, lname in label_names.items():
    mask  = y_test == lid
    f_acc = (preds_float[mask] == lid).mean() * 100
    q_acc = (preds_quant[mask] == lid).mean() * 100
    print(f"  {lname:>12} | {f_acc:>6.2f}% | {q_acc:>8.2f}% | {(f_acc-q_acc):>5.2f}%")

# =============================================================
# STEP 4 — SAVE WEIGHTS
# =============================================================

print("\n[4/5] Saving quantized weights...")
print("-" * 40)

# CSV
rows = []
for name, arr_f, arr_q, scale in [
    ('W1',W1,W1_q,s_W1),('b1',b1.reshape(1,-1),b1_q.reshape(1,-1),s_b1),
    ('W2',W2,W2_q,s_W2),('b2',b2.reshape(1,-1),b2_q.reshape(1,-1),s_b2),
    ('W3',W3,W3_q,s_W3),('b3',b3.reshape(1,-1),b3_q.reshape(1,-1),s_b3),
]:
    for i,(fv,qv) in enumerate(zip(arr_f.flatten(), arr_q.flatten())):
        rows.append({'layer':name,'index':i,'float_val':round(float(fv),6),
                     'quant_val':int(qv),'scale':round(scale,4)})
pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR,'weights_quantized.csv'),index=False)
print(f"  Saved: weights_quantized.csv")

# .mem files for Verilog
def save_mem(filename, *arrays):
    with open(os.path.join(OUTPUT_DIR, filename), 'w') as f:
        f.write(f"// {filename} — 8-bit quantized weights (dynamic scale)\n")
        f.write(f"// Load: $readmemh(\"{filename}\", rom);\n\n")
        for arr in arrays:
            for val in arr.flatten():
                f.write(f"{int(val) & 0xFF:02x}\n")
    print(f"  Saved: {filename}")

save_mem('weights_layer1.mem', W1_q, b1_q)
save_mem('weights_layer2.mem', W2_q, b2_q)
save_mem('weights_layer3.mem', W3_q, b3_q)

# Human readable summary
with open(os.path.join(OUTPUT_DIR,'weights_summary.txt'),'w') as f:
    f.write("Quantized Weights Summary\n")
    f.write(f"Bit width: {BIT_WIDTH} | Dynamic per-layer scaling\n")
    f.write("="*60+"\n\n")
    for name, arr_f, arr_q, scale in [
        ('W1',W1,W1_q,s_W1),('b1',b1,b1_q,s_b1),
        ('W2',W2,W2_q,s_W2),('b2',b2,b2_q,s_b2),
        ('W3',W3,W3_q,s_W3),('b3',b3,b3_q,s_b3),
    ]:
        f.write(f"{name} | shape={arr_f.shape} | scale={scale:.4f}\n")
        f.write(f"  Float:     {np.array2string(arr_f.flatten(), precision=4)}\n")
        f.write(f"  Quantized: {arr_q.flatten()}\n\n")
print(f"  Saved: weights_summary.txt")

# =============================================================
# STEP 5 — PLOTS
# =============================================================

print("\n[5/5] Generating plots...")
print("-" * 40)

all_f = np.concatenate([W1.flatten(),b1,W2.flatten(),b2,W3.flatten(),b3])
all_q = np.concatenate([W1_q.flatten(),b1_q,W2_q.flatten(),b2_q,W3_q.flatten(),b3_q])
all_dq = np.concatenate([W1_q.flatten()/s_W1, b1_q/s_b1,
                          W2_q.flatten()/s_W2, b2_q/s_b2,
                          W3_q.flatten()/s_W3, b3_q/s_b3])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('8-bit Dynamic Quantization Analysis', fontsize=13, fontweight='bold')

axes[0].hist(all_f,  bins=30, color='#2196F3', alpha=0.7)
axes[0].set_title('Float32 Weights'); axes[0].set_xlabel('Value')
axes[0].set_ylabel('Count'); axes[0].grid(True, alpha=0.3)

axes[1].hist(all_q, bins=30, color='#F44336', alpha=0.7)
axes[1].set_title('Quantized Weights (8-bit int)'); axes[1].set_xlabel('Value')
axes[1].set_ylabel('Count'); axes[1].grid(True, alpha=0.3)

axes[2].scatter(all_f, all_dq, alpha=0.6, s=20, color='#4CAF50')
axes[2].plot([all_f.min(),all_f.max()],[all_f.min(),all_f.max()],'r--',linewidth=1,label='Ideal')
axes[2].set_title('Float vs Dequantized'); axes[2].set_xlabel('Original Float32')
axes[2].set_ylabel('Dequantized'); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR,'quantization_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: quantization_analysis.png")

# =============================================================
# DONE
# =============================================================

print("\n" + "="*60)
print("QUANTIZATION COMPLETE")
print("="*60)
print(f"  Float32 accuracy : {acc_float:.2f}%")
print(f"  8-bit   accuracy : {acc_quant:.2f}%")
print(f"  Accuracy drop    : {acc_drop:.2f}%")
print(f"\n  Outputs in 'quantization_output/':")
print("    weights_layer1.mem     — Layer 1 for Verilog $readmemh")
print("    weights_layer2.mem     — Layer 2 for Verilog $readmemh")
print("    weights_layer3.mem     — Layer 3 for Verilog $readmemh")
print("    weights_quantized.csv  — All 104 weights as integers")
print("    layer_scales.csv       — Per-layer scale factors")
print("    weights_summary.txt    — Human readable values")
print("    quantization_analysis.png")
print("="*60)