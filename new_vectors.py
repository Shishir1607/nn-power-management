import pandas as pd
import numpy as np

files = ['idle.csv', 'light.csv', 'medium.csv', 'heavy.csv', 'burst.csv']
dfs = []
for f in files:
    df = pd.read_csv(f, encoding='latin-1', on_bad_lines='skip')
    # Drop any rows where the frequency column contains the column name (duplicate headers)
    df = df[df['Average Effective Clock [MHz]'] != 'Average Effective Clock [MHz]']
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract raw columns
freq  = pd.to_numeric(df['Average Effective Clock [MHz]'], errors='coerce').values
temp  = pd.to_numeric(df['CPU (Tctl/Tdie) [°C]'],         errors='coerce').values
usage = pd.to_numeric(df['Total CPU Usage [%]'],           errors='coerce').values
power = pd.to_numeric(df['CPU Package Power [W]'],         errors='coerce').values

# Drop rows with NaN
mask = ~(np.isnan(freq) | np.isnan(temp) | np.isnan(usage) | np.isnan(power))
freq=freq[mask]; temp=temp[mask]; usage=usage[mask]; power=power[mask]
print(f"Clean rows: {len(freq)}")

# Normalization ranges
freq_min,  freq_max  = 115.30,  3955.10
temp_min,  temp_max  = 50.00,   94.10
usage_min, usage_max = 1.20,    100.00
power_min, power_max = 4.10,    39.66

freq_norm  = np.clip((freq  - freq_min)  / (freq_max  - freq_min),  0, 1)
temp_norm  = np.clip((temp  - temp_min)  / (temp_max  - temp_min),  0, 1)
usage_norm = np.clip((usage - usage_min) / (usage_max - usage_min), 0, 1)
power_norm = np.clip((power - power_min) / (power_max - power_min), 0, 1)

switching_activity = power_norm
timing_slack       = (1 - usage_norm) * (1 - temp_norm)

# Labels
labels = []
for i in range(len(freq_norm)):
    u = usage_norm[i]; s = switching_activity[i]
    if   u < 0.08 and s < 0.20: labels.append(0)
    elif u < 0.38 and s < 0.60: labels.append(1)
    elif u < 0.72 and s < 0.78: labels.append(2)
    else:                        labels.append(3)
labels = np.array(labels)

# Scale to 1000x fixed-point
X_fp = np.round(np.stack([freq_norm, temp_norm, usage_norm,
                           switching_activity, timing_slack], axis=1) * 1000).astype(np.int64)

# Weights
W1 = np.array([[-204,834,-676,-1250,1566],[729,-275,975,2069,335],[-106,465,-212,-1499,1485],[285,47,497,1124,324],[315,-218,406,1833,-8],[-206,-126,-269,42,-442],[-80,4,-186,-1497,1179],[393,-199,572,1308,464]],dtype=np.int64)
b1 = np.array([1549626,-76712,1446509,-165209,152380,-195509,1537917,-89662],dtype=np.int64)
W2 = np.array([[180,-215,-350,-137,-271,290,102,146],[-41,-6,162,-291,22,-241,-19,-122],[1766,-1378,2416,-624,-1240,-211,2898,-974],[284,-307,-407,-277,-238,143,70,264]],dtype=np.int64)
b2 = np.array([-182584,-330481,1189913,-198441],dtype=np.int64)
W3 = np.array([[303,-37,1282,-343],[-252,58,784,-172],[298,278,-513,-268],[458,-248,-6086,-487]],dtype=np.int64)
b3 = np.array([-7578354,-1792502,3147357,5905848],dtype=np.int64)

def relu_fp(x): return np.where(x<=0,0,x>>10).astype(np.int64)

names = ['Sleep','LowPower','Balanced','Performance']
correct = 0
per_class = [0,0,0,0]
per_class_total = [0,0,0,0]
total = len(X_fp)

with open('fixedpoint_output/testvectors_real.txt','w') as f:
    for i in range(total):
        xi = X_fp[i]
        h1 = relu_fp(W1@xi + b1)
        h2 = relu_fp(W2@h1 + b2)
        z3 = W3@h2 + b3
        pred = int(np.argmax(z3))
        lab  = int(labels[i])
        per_class_total[lab] += 1
        if pred == lab:
            correct += 1
            per_class[lab] += 1
        f.write(f"{xi[0]} {xi[1]} {xi[2]} {xi[3]} {xi[4]} {lab}\n")

print("="*50)
print(f"  Fixed-Point Accuracy — {total} Real CSV Samples")
print("="*50)
for i in range(4):
    if per_class_total[i] > 0:
        print(f"  {names[i]:12s}: {per_class[i]:4d}/{per_class_total[i]:4d}  ({per_class[i]*100//per_class_total[i]}%)")
print("-"*50)
print(f"  Overall      : {correct:4d}/{total:4d}  ({correct*100//total}%)")
print("="*50)
print(f"\nSaved: fixedpoint_output/testvectors_real.txt")
