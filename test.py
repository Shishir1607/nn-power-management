import numpy as np
import torch
import torch.nn as nn
import os

os.makedirs('fixedpoint_output', exist_ok=True)

class PowerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,8), nn.ReLU(),
            nn.Linear(8,4), nn.ReLU(),
            nn.Linear(4,4)
        )
    def forward(self, x):
        return self.net(x)

model = PowerMLP()
model.load_state_dict(torch.load('training_output/best_model.pth'))
model.eval()

W1 = np.round(model.net[0].weight.data.numpy() * 1000).astype(np.int64)
b1 = np.round(model.net[0].bias.data.numpy()   * 1000 * 1000).astype(np.int64)
W2 = np.round(model.net[2].weight.data.numpy() * 1000).astype(np.int64)
b2 = np.round(model.net[2].bias.data.numpy()   * 1000 * 1000).astype(np.int64)
W3 = np.round(model.net[4].weight.data.numpy() * 1000).astype(np.int64)
b3 = np.round(model.net[4].bias.data.numpy()   * 1000 * 1000).astype(np.int64)

print("W1:", W1.tolist())
print("b1:", b1.tolist())
print("W2:", W2.tolist())
print("b2:", b2.tolist())
print("W3:", W3.tolist())
print("b3:", b3.tolist())

def relu_fp(x):
    return np.where(x <= 0, 0, x >> 10).astype(np.int64)

X_test = np.load('dataset_output/X_test.npy')
y_test = np.load('dataset_output/y_test.npy')
names  = ['Sleep','LowPower','Balanced','Performance']

# Full accuracy
preds = []
for x in X_test:
    xi = np.round(x * 1000).astype(np.int64)
    h1 = relu_fp(W1 @ xi + b1)
    h2 = relu_fp(W2 @ h1 + b2)
    z3 = W3 @ h2 + b3
    preds.append(int(np.argmax(z3)))
preds = np.array(preds)
acc = (preds == y_test).mean() * 100
print(f"\nFixed-point accuracy: {acc:.2f}%")
for i,n in enumerate(names):
    m = y_test==i
    print(f"  {n}: {(preds[m]==i).mean()*100:.2f}%")

# 20 verified test vectors (5 per class)
vectors = []
counts  = [0,0,0,0]
for i in range(len(y_test)):
    lab = int(y_test[i])
    if preds[i]==lab and counts[lab]<5:
        xi = np.round(X_test[i]*1000).astype(int)
        vectors.append((xi.tolist(), lab))
        counts[lab] += 1
    if sum(counts)==20:
        break

print("\nTest vectors:")
for x,lab in vectors:
    print(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]} {lab}")

with open('fixedpoint_output/testvectors.txt','w') as f:
    for x,lab in vectors:
        f.write(f"{x[0]} {x[1]} {x[2]} {x[3]} {x[4]} {lab}\n")

print("\nSaved: fixedpoint_output/testvectors.txt")
