import numpy as np
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- 3. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- 4. Train the kNN classifier (k=3) ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --- 5. Predict on train & test sets ---
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# --- 6. Evaluation Metrics ---
def evaluate(y_true, y_pred, title):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1'])
    
    print(f"\n🧪 {title} Accuracy: {acc:.4f}")
    print(f"📋 {title} Classification Report:\n{report}")
    print(f"🔢 {title} Confusion Matrix:\n{cm}")
    
    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# --- 7. Evaluate on Train and Test sets ---
evaluate(y_train, y_train_pred, "Train")
evaluate(y_test, y_test_pred, "Test")
