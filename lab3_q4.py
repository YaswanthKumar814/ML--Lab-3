import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# --- 1. Load dataset and flatten images ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten the image
])

dataset = datasets.ImageFolder(r'C:\Users\year3\Documents\yaswanthCSE23245\Dataset', transform=transform)

# --- 2. Extract feature vectors (X) and class labels (y) ---
X = []
y = []

for img_tensor, label in dataset:
    X.append(img_tensor.numpy())  # flatten image to numpy vector
    y.append(label)

X = np.array(X)  # shape: (1200, 21168)
y = np.array(y)  # shape: (1200,)

# --- 3. Split into Train and Test sets (70% train, 30% test) ---
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# --- 4. Print sizes for verification ---
print("Total samples:", len(X))
print("Train samples:", len(X_train))
print("Test samples:", len(X_test))

print("Class distribution in Train set:", np.bincount(y_train))
print("Class distribution in Test set:", np.bincount(y_test))
