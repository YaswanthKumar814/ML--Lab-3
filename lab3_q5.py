import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Load and preprocess the dataset ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten the image
])

dataset = datasets.ImageFolder(r'C:\Users\year3\Documents\yaswanthCSE23245\Dataset', transform=transform)

# --- 2. Extract features and labels ---
X = []
y = []

for img_tensor, label in dataset:
    X.append(img_tensor.numpy())
    y.append(label)

X = np.array(X)
y = np.array(y)

# --- 3. Split into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 4. Train kNN classifier with k=3 ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --- 5. Predict on test set ---
y_pred = knn.predict(X_test)

# --- 6. Evaluate the model ---
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))
