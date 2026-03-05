# =============================================================
# Neural Network Based Adaptive Power Management
# Phase 3: MLP Training
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================
# STEP 1 — CONFIGURATION
# =============================================================

DATA_DIR    = 'dataset_output'
OUTPUT_DIR  = 'training_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEARNING_RATE = 0.001
BATCH_SIZE    = 64
MAX_EPOCHS    = 100
PATIENCE      = 10
RANDOM_SEED   = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

label_names = {0: 'Sleep', 1: 'Low Power', 2: 'Balanced', 3: 'Performance'}

print("=" * 60)
print("Power Management MLP Training")
print("=" * 60)

# =============================================================
# STEP 2 — LOAD DATA
# =============================================================

print("\n[1/5] Loading dataset...")
print("-" * 40)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
X_val   = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

print(f"  Train : {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"  Val   : {X_val.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")

# Load class weights
weights_df = pd.read_csv(os.path.join(DATA_DIR, 'class_weights.csv'))
class_weights = torch.tensor(weights_df['weight'].values, dtype=torch.float32)
print(f"\n  Class weights loaded:")
for _, row in weights_df.iterrows():
    print(f"    [{int(row['class_id'])}] {row['class_name']:12s}: {row['weight']:.4f}")

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE, shuffle=True)

val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE, shuffle=False)

# =============================================================
# STEP 3 — BUILD MODEL
# =============================================================

print("\n[2/5] Building MLP model...")
print("-" * 40)

class PowerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 4)
        )

    def forward(self, x):
        return self.net(x)

model = PowerMLP()

total_params = sum(p.numel() for p in model.parameters())
print(f"  Architecture : 5 → 8 → 4 → 4")
print(f"  Total params : {total_params}")
print(f"  Model:\n{model}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================================================
# STEP 4 — TRAINING LOOP
# =============================================================

print("\n[3/5] Training...")
print("-" * 40)
print(f"  Epochs: {MAX_EPOCHS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE} | Patience: {PATIENCE}")
print()
print(f"  {'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Status'}")
print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}")

train_losses, val_losses = [], []
train_accs,   val_accs   = [], []

best_val_acc     = 0.0
best_val_loss    = float('inf')
patience_counter = 0
best_epoch       = 0

for epoch in range(1, MAX_EPOCHS + 1):

    # --- Training ---
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total   = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * len(y_batch)
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == y_batch).sum().item()
        train_total   += len(y_batch)

    train_loss /= train_total
    train_acc   = train_correct / train_total * 100

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            val_loss    += loss.item() * len(y_batch)
            preds        = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total   += len(y_batch)

    val_loss /= val_total
    val_acc   = val_correct / val_total * 100

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # --- Early stopping ---
    status = ''
    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_val_loss = val_loss
        best_epoch    = epoch
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
        status = '✓ saved'
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  {epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>6.2f}% | Early stop")
            break

    if epoch % 5 == 0 or epoch <= 5 or status:
        print(f"  {epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>6.2f}% | {status}")

print()
print(f"  Best model: Epoch {best_epoch} | Val Acc = {best_val_acc:.2f}% | Val Loss = {best_val_loss:.4f}")

# =============================================================
# STEP 5 — EVALUATE ON TEST SET
# =============================================================

print("\n[4/5] Evaluating on test set...")
print("-" * 40)

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256):
        outputs = model(X_batch)
        preds   = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(y_batch.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean() * 100
print(f"  Overall Test Accuracy: {test_acc:.2f}%")

print()
print(f"  Per-class results:")
print(f"  {'Class':>12} | {'Total':>6} | {'Correct':>8} | {'Accuracy':>8}")
print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}")
for label_id, label_name in label_names.items():
    mask    = all_labels == label_id
    total   = mask.sum()
    correct = (all_preds[mask] == label_id).sum()
    acc     = correct / total * 100 if total > 0 else 0
    print(f"  {label_name:>12} | {total:>6} | {correct:>8} | {acc:>7.2f}%")

# Confusion matrix
print()
print("  Confusion Matrix (rows=actual, cols=predicted):")
print(f"  {'':>12}", end='')
for name in label_names.values():
    print(f"  {name[:8]:>8}", end='')
print()
for actual_id, actual_name in label_names.items():
    print(f"  {actual_name:>12}", end='')
    for pred_id in range(4):
        mask  = all_labels == actual_id
        count = (all_preds[mask] == pred_id).sum()
        print(f"  {count:>8}", end='')
    print()

# Save confusion matrix as CSV
cm_data = {}
for pred_id, pred_name in label_names.items():
    col = []
    for actual_id in range(4):
        mask  = all_labels == actual_id
        count = (all_preds[mask] == pred_id).sum()
        col.append(count)
    cm_data[pred_name] = col

cm_df = pd.DataFrame(cm_data, index=list(label_names.values()))
cm_df.to_csv(os.path.join(OUTPUT_DIR, 'confusion_matrix.csv'))

# =============================================================
# STEP 6 — PLOTS
# =============================================================

print("\n[5/5] Generating plots...")
print("-" * 40)

epochs_ran = len(train_losses)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('MLP Training Results — Power Management', fontsize=13, fontweight='bold')

# Loss curve
axes[0].plot(range(1, epochs_ran+1), train_losses, label='Train Loss', color='#2196F3', linewidth=2)
axes[0].plot(range(1, epochs_ran+1), val_losses,   label='Val Loss',   color='#F44336', linewidth=2)
axes[0].axvline(best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training & Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy curve
axes[1].plot(range(1, epochs_ran+1), train_accs, label='Train Acc', color='#2196F3', linewidth=2)
axes[1].plot(range(1, epochs_ran+1), val_accs,   label='Val Acc',   color='#F44336', linewidth=2)
axes[1].axvline(best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
axes[1].axhline(92, color='orange', linestyle=':', alpha=0.7, label='92% target')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].set_title('Training & Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: training_curves.png")

# Confusion matrix heatmap
fig, ax = plt.subplots(figsize=(8, 6))
cm_array = np.zeros((4, 4), dtype=int)
for actual_id in range(4):
    for pred_id in range(4):
        mask = all_labels == actual_id
        cm_array[actual_id][pred_id] = (all_preds[mask] == pred_id).sum()

im = ax.imshow(cm_array, interpolation='nearest', cmap='Blues')
plt.colorbar(im)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(list(label_names.values()), rotation=45, ha='right')
ax.set_yticklabels(list(label_names.values()))
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title(f'Confusion Matrix — Test Accuracy: {test_acc:.2f}%')

for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm_array[i][j]),
                ha='center', va='center',
                color='white' if cm_array[i][j] > cm_array.max()/2 else 'black',
                fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: confusion_matrix.png")

# =============================================================
# DONE
# =============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Best Validation Accuracy : {best_val_acc:.2f}%")
print(f"  Test Accuracy            : {test_acc:.2f}%")
print(f"  Best Epoch               : {best_epoch}")
print(f"\nOutputs saved to '{OUTPUT_DIR}/' folder:")
print("  best_model.pth       — Trained model weights")
print("  training_curves.png  — Loss and accuracy plots")
print("  confusion_matrix.png — Confusion matrix heatmap")
print("  confusion_matrix.csv — Confusion matrix data")
if test_acc >= 92:
    print(f"\n  ✓ TARGET MET — {test_acc:.2f}% >= 92%")
else:
    print(f"\n  ✗ Below 92% target ({test_acc:.2f}%) — share output and I'll adjust")
print("=" * 60)