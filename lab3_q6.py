import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# --- 1. Load and preprocess the MiniImageNet (2-class) dataset ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),               # Resize to match MiniImageNet format
    transforms.ToTensor(),                     # Convert PIL Image to tensor
    transforms.Lambda(lambda x: x.view(-1))    # Flatten image to 1D
])

# Replace path with your actual dataset location
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

# --- 3. Split data into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 4. Train k-NN classifier (k=3) ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --- 5. Predict using test data ---
y_pred = knn.predict(X_test)

# --- 6. Evaluate the model ---
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1'])
conf_matrix = confusion_matrix(y_test, y_pred)

# --- 7. Print Evaluation Results ---
print("✅ Test Accuracy: {:.2f}".format(accuracy))
print("\n📋 Classification Report:\n", report)
print("\n🔢 Confusion Matrix:\n", conf_matrix)

# --- 8. Optional: Visualize Confusion Matrix ---
import seaborn as sns

plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
