import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- 1. Load the dataset (assuming same setup as A1) ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten
])

dataset = datasets.ImageFolder(r'C:\Users\year3\Documents\yaswanthCSE23245\Dataset', transform=transform)

# --- 2. Extract all features into a numpy array ---
all_features = []
for img, label in dataset:
    all_features.append(img.numpy())

all_features = np.array(all_features)  # shape: (num_images, 21168)

# --- 3. Select one feature (e.g., feature index 1000) ---
feature_index = 1000
selected_feature_values = all_features[:, feature_index]  # shape: (num_images,)

# --- 4. Calculate Mean and Variance ---
mean = np.mean(selected_feature_values)
variance = np.var(selected_feature_values)

print(f"Feature Index: {feature_index}")
print(f"Mean: {mean}")
print(f"Variance: {variance}")

# --- 5. Plot Histogram using Buckets ---
plt.hist(selected_feature_values, bins=20, edgecolor='black')
plt.title(f'Histogram of Feature Index {feature_index}')
plt.xlabel('Pixel Intensity (normalized)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

