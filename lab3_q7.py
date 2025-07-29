import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Load and preprocess the dataset ---
transform = transforms.Compose([
    transforms.Resize((84, 84)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

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

# --- 3. Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 4. Train kNN classifier ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --- 5. Test predictions ---
y_pred = knn.predict(X_test)

# --- 6. Accuracy and evaluation ---
accuracy = accuracy_score(y_test, y_pred)
print("✅ Test Accuracy: {:.2f}".format(accuracy))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔢 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- 7. Visualize Confusion Matrix ---
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# =====================
# === A7 Predictions ===
# =====================

# --- 8. Predict on ALL test samples ---
print("\n🔍 First 10 Predictions vs Actual Labels:")
for i in range(10):
    prediction = knn.predict([X_test[i]])[0]
    actual = y_test[i]
    print(f"Test Sample {i}: Predicted = Class {prediction}, Actual = Class {actual}")

# --- 9. Predict a single test vector manually ---
print("\n🎯 Predicting one test vector directly:")
test_vector = X_test[50]  # pick any index
predicted_class = knn.predict([test_vector])[0]
print(f"Prediction for test sample 50 → Class {predicted_class}, Actual → Class {y_test[50]}")
