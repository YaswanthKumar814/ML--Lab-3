import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- 1. Load and preprocess the MiniImageNet (2-class) dataset ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Load dataset
dataset = datasets.ImageFolder(
    r'C:\yaswanth\BTECH\Sem 5\Machine Learning\Lab\Dataset',
    transform=transform
)

# --- 2. Extract features and labels ---
X = []
y = []

for img_tensor, label in dataset:
    X.append(img_tensor.numpy())
    y.append(label)

X = np.array(X)
y = np.array(y)

# --- 3. Split into train/test sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 4. Try different values of k ---
k_values = list(range(1, 12))  # k = 1 to 11
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"k = {k}, Accuracy = {acc:.4f}")

# --- 5. Plot Accuracy vs k ---
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-')
plt.title('kNN Accuracy vs k (MiniImageNet 2-class)')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()
