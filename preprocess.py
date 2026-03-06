# =============================================================
# Neural Network Based Adaptive Power Management
# Phase 2: Data Preprocessing
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================================================
# CONFIGURATION
# =============================================================

CSV_FILES = {
    'idle'  : 'idle.CSV',
    'light' : 'light.CSV',
    'medium': 'medium.CSV',
    'heavy' : 'heavy.CSV',
    'burst' : 'burst.CSV',
}

OUTPUT_DIR     = 'dataset_output'
TARGET_SAMPLES = 20000
NOISE_STD      = 0.02
RANDOM_SEED    = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)

label_names = {0: 'Sleep', 1: 'Low Power', 2: 'Balanced', 3: 'Performance'}

print("=" * 60)
print("Power Management Dataset Preprocessing")
print("=" * 60)

# =============================================================
# STEP 1 — LOAD CSV FILES
# =============================================================

print("\n[1/6] Loading CSV files...")
print("-" * 40)

def load_hwinfo_csv(filepath):
    """
    HWiNFO repeats the header row mid-file every few hundred rows.
    This function handles that by reading line by line and
    stitching chunks together.
    """
    chunks  = []
    header  = None
    rows    = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            cols = line.split(',')
            if cols[0].strip().lower() in ['date', '"date"', 'date"']:
                if rows and header is not None:
                    chunks.append(pd.DataFrame(rows, columns=header))
                    rows = []
                header = [c.strip().strip('"') for c in cols]
            else:
                if header is not None:
                    rows.append([c.strip().strip('"') for c in cols[:len(header)]])

    if rows and header is not None:
        chunks.append(pd.DataFrame(rows, columns=header))

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

all_sessions = []

for session_name, filename in CSV_FILES.items():
    if not os.path.exists(filename):
        print(f"  ⚠ {filename} not found — skipping")
        continue
    df = load_hwinfo_csv(filename)
    df['session'] = session_name
    all_sessions.append(df)
    print(f"  {session_name:8s} →  {len(df)} rows loaded from {filename}")

combined = pd.concat(all_sessions, ignore_index=True)

# =============================================================
# STEP 2 — EXTRACT FEATURES
# =============================================================

print("\n[2/6] Extracting features...")
print("-" * 40)

# Column name patterns to search for
COL_PATTERNS = {
    'freq_mhz'     : ['average effective clock', 'effective clock'],
    'temp_c'       : ['cpu (tctl/tdie)', 'tctl/tdie', 'cpu die temp'],
    'cpu_usage_pct': ['total cpu usage'],
    'pkg_power_w'  : ['cpu package power'],
}

def find_column(df, patterns):
    """Find first column matching any of the given patterns (case-insensitive)."""
    for col in df.columns:
        col_lower = col.lower()
        for p in patterns:
            if p in col_lower:
                return col
    return None

extracted_frames = []

for session_name in CSV_FILES.keys():
    df = combined[combined['session'] == session_name].copy()
    if df.empty:
        continue

    row = {}
    valid = True

    for feat, patterns in COL_PATTERNS.items():
        col = find_column(df, patterns)
        if col is None:
            print(f"  ⚠ Could not find column for '{feat}' in {session_name}")
            valid = False
            break
        df[feat] = pd.to_numeric(df[col], errors='coerce')
        row[feat] = col

    if not valid:
        continue

    feat_cols  = list(COL_PATTERNS.keys())
    df_clean   = df[feat_cols + ['session']].dropna()
    df_clean   = df_clean.reset_index(drop=True)

    print(f"  {session_name:8s} →  {len(df_clean)} clean rows | "
          f"CPU: {df_clean['cpu_usage_pct'].astype(float).mean():.1f}% | "
          f"Temp: {df_clean['temp_c'].astype(float).mean():.1f}°C | "
          f"Power: {df_clean['pkg_power_w'].astype(float).mean():.1f}W")

    extracted_frames.append(df_clean)

combined = pd.concat(extracted_frames, ignore_index=True)

# Convert all feature columns to float
for col in COL_PATTERNS.keys():
    combined[col] = pd.to_numeric(combined[col], errors='coerce')

combined = combined.dropna().reset_index(drop=True)
print(f"\n  Total rows after extraction: {len(combined)}")

# =============================================================
# STEP 3 — NORMALIZE & DERIVE FEATURES
# =============================================================

print("\n[3/6] Deriving switching activity and timing slack...")
print("-" * 40)

# Compute min/max from data
feature_ranges = {
    'freq_mhz'     : (combined['freq_mhz'].min(),      combined['freq_mhz'].max()),
    'temp_c'       : (combined['temp_c'].min(),         combined['temp_c'].max()),
    'cpu_usage_pct': (combined['cpu_usage_pct'].min(),  combined['cpu_usage_pct'].max()),
    'pkg_power_w'  : (combined['pkg_power_w'].min(),    combined['pkg_power_w'].max()),
}

def normalize(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val + 1e-8)

combined['freq_norm']  = normalize(combined['freq_mhz'],      *feature_ranges['freq_mhz'])
combined['temp_norm']  = normalize(combined['temp_c'],         *feature_ranges['temp_c'])
combined['usage_norm'] = normalize(combined['cpu_usage_pct'],  *feature_ranges['cpu_usage_pct'])
combined['power_norm'] = normalize(combined['pkg_power_w'],    *feature_ranges['pkg_power_w'])

# Clip to [0, 1]
for col in ['freq_norm', 'temp_norm', 'usage_norm', 'power_norm']:
    combined[col] = combined[col].clip(0, 1)

# Save feature stats for FPGA normalization (no voltage)
stats = {
    'freq_min':  feature_ranges['freq_mhz'][0],
    'freq_max':  feature_ranges['freq_mhz'][1],
    'temp_min':  feature_ranges['temp_c'][0],
    'temp_max':  feature_ranges['temp_c'][1],
    'usage_min': feature_ranges['cpu_usage_pct'][0],
    'usage_max': feature_ranges['cpu_usage_pct'][1],
    'power_min': feature_ranges['pkg_power_w'][0],
    'power_max': feature_ranges['pkg_power_w'][1],
}

print(f"  Feature ranges (save these for FPGA normalization):")
for k, v in stats.items():
    print(f"    {k:16s} = {v:.4f}")

pd.DataFrame([stats]).T.rename(columns={0: 'value'}).to_csv(
    os.path.join(OUTPUT_DIR, 'feature_stats.csv'))
print(f"  Saved to {OUTPUT_DIR}/feature_stats.csv")

# Derive switching activity and timing slack
combined['switching_activity'] = combined['power_norm']
combined['timing_slack']       = (1 - combined['usage_norm']) * (1 - combined['temp_norm'])

print(f"\n  switching_activity: mean={combined['switching_activity'].mean():.3f}, "
      f"std={combined['switching_activity'].std():.3f}")
print(f"  timing_slack:       mean={combined['timing_slack'].mean():.3f}, "
      f"std={combined['timing_slack'].std():.3f}")

# Final 5 features — no voltage
FEATURE_COLS = ['freq_norm', 'temp_norm', 'usage_norm',
                'switching_activity', 'timing_slack']

# =============================================================
# STEP 4 — LABEL
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

# Class distribution before augmentation
print(f"  Class distribution before augmentation:")
total = len(combined)
for lid, lname in label_names.items():
    count = (combined['label'] == lid).sum()
    pct   = count / total * 100
    bar   = '█' * int(pct / 2)
    print(f"    [{lid}] {lname:12s} : {count:5d} ({pct:2.0f}%) {bar}")

# Class weights
print(f"\n  Class weights for weighted loss (saved to class_weights.csv):")
weights = {}
for lid, lname in label_names.items():
    count      = (combined['label'] == lid).sum()
    weight     = total / (4 * count) if count > 0 else 1.0
    weights[lid] = weight
    print(f"    [{lid}] {lname:12s} : weight = {weight:.4f}")

weights_df = pd.DataFrame([
    {'class_id': lid, 'class_name': lname, 'count': (combined['label'] == lid).sum(), 'weight': weights[lid]}
    for lid, lname in label_names.items()
])
weights_df.to_csv(os.path.join(OUTPUT_DIR, 'class_weights.csv'), index=False)
print(f"\n  Using natural class distribution (imbalance handled by weighted loss in train.py)")

# =============================================================
# STEP 5 — AUGMENT
# =============================================================

print("\n[5/6] Augmenting dataset...")
print("-" * 40)

X_real = combined[FEATURE_COLS].values.astype(np.float32)
y_real = combined['label'].values.astype(np.int64)

current_size = len(X_real)
need         = TARGET_SAMPLES - current_size
print(f"  Current size: {current_size} | Target: {TARGET_SAMPLES} | Generating: {need} augmented samples")

aug_indices = np.random.choice(current_size, size=need, replace=True)
X_aug = X_real[aug_indices] + np.random.normal(0, NOISE_STD, (need, X_real.shape[1])).astype(np.float32)
X_aug = np.clip(X_aug, 0, 1)

# Re-label augmented samples
aug_df = pd.DataFrame(X_aug, columns=FEATURE_COLS)
aug_df['usage_norm']         = aug_df['usage_norm']
aug_df['switching_activity'] = aug_df['switching_activity']
y_aug = aug_df.apply(label_sample, axis=1).values.astype(np.int64)

X_all = np.vstack([X_real, X_aug])
y_all = np.concatenate([y_real, y_aug])

print(f"  Final dataset size: {len(X_all)}")
print(f"\n  Class distribution after augmentation:")
for lid, lname in label_names.items():
    count = (y_all == lid).sum()
    pct   = count / len(y_all) * 100
    bar   = '█' * int(pct / 2)
    print(f"    [{lid}] {lname:12s} : {count:5d} ({pct:2.0f}%) {bar}")

# =============================================================
# STEP 6 — SPLIT & SAVE
# =============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=RANDOM_SEED, stratify=y_all)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=RANDOM_SEED, stratify=y_temp)

print(f"\n  Split sizes:")
print(f"    Train :  {len(X_train)} samples (70%)")
print(f"    Val   :  {len(X_val)} samples (15%)")
print(f"    Test  :  {len(X_test)} samples (15%)")

np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'),   X_val)
np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'),   y_val)
np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'),  X_test)
np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'),  y_test)
print(f"\n  Saved all splits to {OUTPUT_DIR}/")

# Full dataset CSV
full_df = pd.DataFrame(X_all, columns=FEATURE_COLS)
full_df['label'] = y_all
full_df.to_csv(os.path.join(OUTPUT_DIR, 'full_dataset.csv'), index=False)

# =============================================================
# STEP 7 — PLOTS
# =============================================================

print("\n[6/6] Generating plots...")
print("-" * 40)

colors = {0: '#00d4ff', 1: '#00ff9d', 2: '#ffd166', 3: '#ff6b35'}

# Feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Feature Distributions by Power Mode', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, feat in enumerate(FEATURE_COLS):
    ax = axes[i]
    for lid, lname in label_names.items():
        mask = y_all == lid
        ax.hist(X_all[mask, i], bins=40, alpha=0.5,
                color=colors[lid], label=lname, density=True)
    ax.set_title(feat)
    ax.set_xlabel('Normalized Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Hide unused subplot
axes[5].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: feature_distributions.png")

# Usage vs switching activity scatter
fig, ax = plt.subplots(figsize=(10, 7))
for lid, lname in label_names.items():
    mask = y_all == lid
    ax.scatter(X_all[mask, 2], X_all[mask, 3],
               alpha=0.15, s=5, color=colors[lid], label=lname)

# Draw label boundaries
ax.axvline(0.08,  color='white', linewidth=0.8, linestyle='--', alpha=0.4)
ax.axvline(0.38,  color='white', linewidth=0.8, linestyle='--', alpha=0.4)
ax.axvline(0.72,  color='white', linewidth=0.8, linestyle='--', alpha=0.4)
ax.axhline(0.20,  color='cyan',  linewidth=0.8, linestyle=':',  alpha=0.4)
ax.axhline(0.60,  color='cyan',  linewidth=0.8, linestyle=':',  alpha=0.4)
ax.axhline(0.78,  color='cyan',  linewidth=0.8, linestyle=':',  alpha=0.4)

ax.set_xlabel('usage_norm')
ax.set_ylabel('switching_activity (power_norm)')
ax.set_title('Label Boundaries — Usage vs Switching Activity')
ax.legend(markerscale=4)
ax.set_facecolor('#0a0e1a')
fig.patch.set_facecolor('#0a0e1a')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'usage_vs_slack_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: usage_vs_slack_scatter.png")

# Raw CPU usage per session
fig, ax = plt.subplots(figsize=(14, 5))
offset = 0
session_colors = {'idle':'#00d4ff','light':'#00ff9d','medium':'#ffd166',
                  'heavy':'#ff6b35','burst':'#7b4fff'}
for sess, fname in CSV_FILES.items():
    mask = combined['session'] == sess
    vals = combined[mask]['usage_norm'].values
    ax.plot(range(offset, offset + len(vals)), vals,
            color=session_colors.get(sess, 'white'), linewidth=0.8, label=sess)
    ax.axvline(offset, color='white', linewidth=0.5, alpha=0.2)
    offset += len(vals)

ax.set_xlabel('Sample index')
ax.set_ylabel('CPU Usage (normalized)')
ax.set_title('Raw CPU Usage per Session')
ax.legend()
ax.set_facecolor('#0a0e1a')
fig.patch.set_facecolor('#0a0e1a')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.title.set_color('white')
plt.tight_layout()
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