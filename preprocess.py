# =============================================================
# Neural Network Based Adaptive Power Management
# Phase 2: Data Preprocessing & Dataset Creation
# ASUS Vivobook AMD Ryzen 7 5825U
# =============================================================

import os
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split

# =============================================================
# STEP 1 — CONFIGURATION
# =============================================================

CSV_FILES = {
    'idle':   'idle.CSV',
    'light':  'light.CSV',
    'medium': 'medium.CSV',
    'heavy':  'heavy.CSV',
    'burst':  'burst.CSV',
}

TARGET_SAMPLES  = 20000
NOISE_STD       = 0.02
RANDOM_SEED     = 42
OUTPUT_DIR      = 'dataset_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Power Management Dataset Preprocessing")
print("=" * 60)

# =============================================================
# STEP 2 — LOAD CSV FILES
# =============================================================

def load_hwinfo_csv(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    chunks = []
    current_chunk = []
    header = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('Date'):
            if current_chunk and header:
                try:
                    chunks.append(pd.read_csv(io.StringIO(header + ''.join(current_chunk))))
                except Exception:
                    pass
            header = line
            current_chunk = []
        elif stripped and not stripped.startswith(',,'):
            current_chunk.append(line)
    if current_chunk and header:
        try:
            chunks.append(pd.read_csv(io.StringIO(header + ''.join(current_chunk))))
        except Exception:
            pass
    if not chunks:
        raise ValueError(f"Could not parse {filepath}")
    return pd.concat(chunks, ignore_index=True)


def find_column(df, keyword):
    matches = [c for c in df.columns if keyword in c]
    if not matches:
        matches = [c for c in df.columns if 'Tctl' in c or 'Tdie' in c]
    return matches[0] if matches else None


print("\n[1/6] Loading CSV files...")
print("-" * 40)

all_dfs = []
for session_name, filename in CSV_FILES.items():
    if not os.path.exists(filename):
        print(f"  WARNING: {filename} not found — skipping {session_name}")
        continue
    try:
        df = load_hwinfo_csv(filename)
        print(f"  {session_name:8s} → {len(df):5d} rows loaded from {filename}")
        df['session'] = session_name
        all_dfs.append(df)
    except Exception as e:
        print(f"  ERROR loading {filename}: {e}")

if not all_dfs:
    print("ERROR: No CSV files could be loaded.")
    exit(1)

# =============================================================
# STEP 3 — EXTRACT FEATURES
# =============================================================

print("\n[2/6] Extracting features...")
print("-" * 40)

def extract_features(df, session_name):
    extracted = pd.DataFrame()

    col = find_column(df, 'Average Effective Clock')
    if col:
        extracted['freq_mhz'] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"  WARNING: Frequency column not found in {session_name}")
        return None

    temp_col = None
    for c in df.columns:
        if 'Tctl' in c or 'Tdie' in c:
            temp_col = c
            break
    if temp_col:
        extracted['temp_c'] = pd.to_numeric(df[temp_col], errors='coerce')
    else:
        print(f"  WARNING: Temperature column not found in {session_name}")
        return None

    col = find_column(df, 'Total CPU Usage')
    if col:
        extracted['cpu_usage_pct'] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"  WARNING: CPU Usage column not found in {session_name}")
        return None

    col = find_column(df, 'CPU Package Power')
    if col:
        extracted['pkg_power_w'] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"  WARNING: Power column not found in {session_name}")
        return None

    col = find_column(df, 'Core Voltage (SVI2')
    if col:
        extracted['voltage_v'] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"  WARNING: Voltage column not found in {session_name}")
        return None

    extracted['session'] = session_name
    extracted.dropna(inplace=True)
    print(f"  {session_name:8s} → {len(extracted):5d} clean rows | "
          f"CPU: {extracted['cpu_usage_pct'].mean():.1f}% | "
          f"Temp: {extracted['temp_c'].mean():.1f}°C | "
          f"Power: {extracted['pkg_power_w'].mean():.1f}W")
    return extracted

feature_dfs = []
for df in all_dfs:
    session = df['session'].iloc[0]
    feat_df = extract_features(df, session)
    if feat_df is not None:
        feature_dfs.append(feat_df)

combined = pd.concat(feature_dfs, ignore_index=True)
print(f"\n  Total rows after extraction: {len(combined)}")

# =============================================================
# STEP 4 — NORMALIZE + DERIVE FEATURES
# =============================================================

print("\n[3/6] Deriving switching activity and timing slack...")
print("-" * 40)

feature_stats = {
    'freq_min':    combined['freq_mhz'].min(),
    'freq_max':    combined['freq_mhz'].max(),
    'temp_min':    combined['temp_c'].min(),
    'temp_max':    combined['temp_c'].max(),
    'usage_min':   combined['cpu_usage_pct'].min(),
    'usage_max':   combined['cpu_usage_pct'].max(),
    'power_min':   combined['pkg_power_w'].min(),
    'power_max':   combined['pkg_power_w'].max(),
    'voltage_min': combined['voltage_v'].min(),
    'voltage_max': combined['voltage_v'].max(),
}

print("  Feature ranges (save these for FPGA normalization):")
for k, v in feature_stats.items():
    print(f"    {k:15s} = {v:.4f}")

stats_df = pd.DataFrame(list(feature_stats.items()), columns=['parameter', 'value'])
stats_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_stats.csv'), index=False)
print(f"  Saved to {OUTPUT_DIR}/feature_stats.csv")

combined['freq_norm']    = (combined['freq_mhz']     - feature_stats['freq_min'])    / (feature_stats['freq_max']    - feature_stats['freq_min'])
combined['temp_norm']    = (combined['temp_c']        - feature_stats['temp_min'])    / (feature_stats['temp_max']    - feature_stats['temp_min'])
combined['usage_norm']   = (combined['cpu_usage_pct'] - feature_stats['usage_min'])   / (feature_stats['usage_max']   - feature_stats['usage_min'])
combined['power_norm']   = (combined['pkg_power_w']   - feature_stats['power_min'])   / (feature_stats['power_max']   - feature_stats['power_min'])
combined['voltage_norm'] = (combined['voltage_v']     - feature_stats['voltage_min']) / (feature_stats['voltage_max'] - feature_stats['voltage_min'])

combined['switching_activity'] = combined['power_norm']
combined['timing_slack']       = (1.0 - combined['usage_norm']) * (1.0 - combined['temp_norm'])

for col in ['freq_norm','temp_norm','usage_norm','power_norm','voltage_norm','switching_activity','timing_slack']:
    combined[col] = combined[col].clip(0.0, 1.0)

print(f"\n  switching_activity: mean={combined['switching_activity'].mean():.3f}, std={combined['switching_activity'].std():.3f}")
print(f"  timing_slack:       mean={combined['timing_slack'].mean():.3f}, std={combined['timing_slack'].std():.3f}")

# =============================================================
# STEP 5 — LABEL
# =============================================================

print("\n[4/6] Labeling dataset...")
print("-" * 40)

def label_sample(row):
    wl  = row['usage_norm']
    pwr = row['switching_activity']

    if wl < 0.08 and pwr < 0.20:
        return 0   # Sleep
    elif wl < 0.38 and pwr < 0.60:
        return 1   # Low Power
    elif wl < 0.72 and pwr < 0.78:
        return 2   # Balanced
    else:
        return 3   # Performance

combined['label'] = combined.apply(label_sample, axis=1)

label_names = {0: 'Sleep', 1: 'Low Power', 2: 'Balanced', 3: 'Performance'}
print("  Class distribution before augmentation:")
total = len(combined)
class_counts_raw = []
for label_id, label_name in label_names.items():
    count = (combined['label'] == label_id).sum()
    class_counts_raw.append(count)
    pct = count * 100 // total
    bar = '█' * (pct // 2)
    print(f"    [{label_id}] {label_name:12s}: {count:5d} ({pct:3d}%) {bar}")

# Save class weights for train.py
class_weights = [total / (4 * c) if c > 0 else 1.0 for c in class_counts_raw]
weights_df = pd.DataFrame({
    'class_id':   list(range(4)),
    'class_name': list(label_names.values()),
    'count':      class_counts_raw,
    'weight':     class_weights
})
weights_df.to_csv(os.path.join(OUTPUT_DIR, 'class_weights.csv'), index=False)
print(f"\n  Class weights for weighted loss (saved to class_weights.csv):")
for i, (name, w) in enumerate(zip(label_names.values(), class_weights)):
    print(f"    [{i}] {name:12s}: weight = {w:.4f}")

# Using natural distribution — imbalance handled via weighted loss in train.py
print("\n  Using natural class distribution (imbalance handled by weighted loss in train.py)")

# =============================================================
# STEP 6 — AUGMENT
# =============================================================

print("\n[5/6] Augmenting dataset...")
print("-" * 40)

feature_cols = ['freq_norm', 'temp_norm', 'usage_norm', 'switching_activity', 'timing_slack']

np.random.seed(RANDOM_SEED)

current_size = len(combined)
needed = max(0, TARGET_SAMPLES - current_size)
print(f"  Current size: {current_size} | Target: {TARGET_SAMPLES} | Generating: {needed} augmented samples")

augmented_rows = []
for _ in range(needed):
    sample = combined.sample(1, random_state=None).copy()
    noise = np.random.normal(0, NOISE_STD, len(feature_cols))
    row_vals = sample[feature_cols].values[0]
    new_vals = np.clip(row_vals + noise, 0.0, 1.0)
    new_row = sample.copy()
    for i, col in enumerate(feature_cols):
        new_row[col] = new_vals[i]
    new_row['label'] = label_sample(new_row.iloc[0])
    augmented_rows.append(new_row)

augmented_df = pd.concat(augmented_rows, ignore_index=True)
full_dataset  = pd.concat([combined, augmented_df], ignore_index=True)
full_dataset  = full_dataset.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"  Final dataset size: {len(full_dataset)}")
print("\n  Class distribution after augmentation:")
total = len(full_dataset)
for label_id, label_name in label_names.items():
    count = (full_dataset['label'] == label_id).sum()
    pct = count * 100 // total
    bar = '█' * (pct // 2)
    print(f"    [{label_id}] {label_name:12s}: {count:5d} ({pct:3d}%) {bar}")

# =============================================================
# STEP 7 — SPLIT
# =============================================================

X = full_dataset[feature_cols].values.astype(np.float32)
y = full_dataset['label'].values.astype(np.int64)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_SEED)

print(f"\n  Split sizes:")
print(f"    Train : {len(X_train):6d} samples (70%)")
print(f"    Val   : {len(X_val):6d} samples (15%)")
print(f"    Test  : {len(X_test):6d} samples (15%)")

np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'),   X_val)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),   y_val)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'),  X_test)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  y_test)

full_dataset[feature_cols + ['label', 'session']].to_csv(
    os.path.join(OUTPUT_DIR, 'full_dataset.csv'), index=False)

print(f"\n  Saved all splits to {OUTPUT_DIR}/")

# =============================================================
# STEP 8 — PLOTS
# =============================================================

print("\n[6/6] Generating plots...")
print("-" * 40)

colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
label_display = ['Sleep', 'Low Power', 'Balanced', 'Performance']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Power Management Dataset — Feature Distributions', fontsize=14, fontweight='bold')

feature_display_names = {
    'freq_norm':          'Frequency (normalized)',
    'temp_norm':          'Temperature (normalized)',
    'usage_norm':         'CPU Usage (normalized)',
    'switching_activity': 'Switching Activity',
    'timing_slack':       'Timing Slack',
}

for idx, (col, display_name) in enumerate(feature_display_names.items()):
    ax = axes[idx // 3][idx % 3]
    for label_id in range(4):
        subset = full_dataset[full_dataset['label'] == label_id][col]
        ax.hist(subset, bins=50, alpha=0.6, color=colors[label_id],
                label=label_display[label_id], density=True)
    ax.set_title(display_name)
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

ax = axes[1][2]
class_counts = [(full_dataset['label'] == i).sum() for i in range(4)]
bars = ax.bar(label_display, class_counts, color=colors)
ax.set_title('Class Distribution (Natural + Augmented)')
ax.set_ylabel('Sample Count')
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 100,
            str(count), ha='center', va='bottom', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: feature_distributions.png")

fig, ax = plt.subplots(figsize=(10, 6))
for label_id in range(4):
    subset = full_dataset[full_dataset['label'] == label_id]
    ax.scatter(subset['usage_norm'], subset['timing_slack'],
               alpha=0.3, s=5, color=colors[label_id], label=label_display[label_id])
ax.set_xlabel('CPU Usage (normalized)')
ax.set_ylabel('Timing Slack (derived)')
ax.set_title('CPU Usage vs Timing Slack — Label Boundaries')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'usage_vs_slack_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: usage_vs_slack_scatter.png")

fig, ax = plt.subplots(figsize=(12, 5))
session_colors = {'idle': '#2196F3', 'light': '#4CAF50',
                  'medium': '#FF9800', 'heavy': '#F44336', 'burst': '#9C27B0'}
for df in feature_dfs:
    session_name = df['session'].iloc[0]
    subset = df['cpu_usage_pct']
    ax.plot(subset.values, alpha=0.7, linewidth=0.8,
            color=session_colors.get(session_name, 'gray'),
            label=f"{session_name} (mean={subset.mean():.1f}%)")
ax.set_xlabel('Sample Index')
ax.set_ylabel('CPU Usage (%)')
ax.set_title('Raw CPU Usage Per Session')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, 'raw_cpu_usage_per_session.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: raw_cpu_usage_per_session.png")

# =============================================================
# DONE
# =============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
print(f"\nOutputs saved to '{OUTPUT_DIR}/' folder:")
print("  X_train.npy, y_train.npy  — Training data")
print("  X_val.npy,   y_val.npy    — Validation data")
print("  X_test.npy,  y_test.npy   — Test data")
print("  full_dataset.csv          — Full labeled dataset")
print("  feature_stats.csv         — Min/max values (save for FPGA!)")
print("  class_weights.csv         — Weights for weighted loss in train.py")
print("  feature_distributions.png")
print("  usage_vs_slack_scatter.png")
print("  raw_cpu_usage_per_session.png")
print("\nNext step: Run train.py to train the MLP model")
print("=" * 60)