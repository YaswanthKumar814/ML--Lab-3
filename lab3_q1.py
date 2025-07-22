from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from numpy.linalg import norm

# Define a transform to convert images to vectors
transform = transforms.Compose([
    transforms.Resize((84, 84)),  # match miniImageNet size
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten the image
])

dataset = datasets.ImageFolder(r'C:\Users\year3\Documents\yaswanthCSE23245\Dataset', transform=transform)
loader = DataLoader(dataset, batch_size=600, shuffle=False)


# Initialize storage
class_0_feats = []
class_1_feats = []

# Load all data
for img, label in dataset:
    if label == 0:
        class_0_feats.append(img.numpy())
    else:
        class_1_feats.append(img.numpy())

# Convert to NumPy arrays
class_0_feats = np.array(class_0_feats)  # shape: (600, 21168)
class_1_feats = np.array(class_1_feats)

mean_0 = np.mean(class_0_feats, axis=0)
mean_1 = np.mean(class_1_feats, axis=0)


std_0 = np.std(class_0_feats, axis=0)
std_1 = np.std(class_1_feats, axis=0)

# Optional: You can average these vectors to get scalar spread values:
spread_0 = np.mean(std_0)
spread_1 = np.mean(std_1)

interclass_distance = norm(mean_0 - mean_1)

print("Class 0 Mean Vector Shape:", mean_0.shape)
print("Class 1 Mean Vector Shape:", mean_1.shape)
print("Class 0 Spread (Avg Std Dev):", spread_0)
print("Class 1 Spread (Avg Std Dev):", spread_1)
print("Distance Between Class Centroids:", interclass_distance)
