import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# --- 1. Load dataset with flattened vectors ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

dataset = datasets.ImageFolder(r'C:\Users\year3\Documents\yaswanthCSE23245\Dataset', transform=transform)

# --- 2. Select any two feature vectors (e.g., image 0 and image 1) ---
vec1 = dataset[0][0].numpy()  # image tensor → numpy array
vec2 = dataset[1][0].numpy()

# --- 3. Compute Minkowski distances for r from 1 to 10 ---
r_values = list(range(1, 11))
distances = []

for r in r_values:
    dist = np.sum(np.abs(vec1 - vec2) ** r) ** (1 / r)
    distances.append(dist)

# --- 4. Plot the distances ---
plt.plot(r_values, distances, marker='o')
plt.title('Minkowski Distance between Two Feature Vectors (r = 1 to 10)')
plt.xlabel('r (Minkowski Power Parameter)')
plt.ylabel('Distance')
plt.grid(True)
plt.show()

# --- 5. Optional: Print distances ---
for r, d in zip(r_values, distances):
    print(f"r = {r}: Distance = {d:.4f}")
