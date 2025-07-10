import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np

PLOT = True
EPOCHS = 100

X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=42)


if PLOT:
    plt.figure(figsize=(6, 5))
    for class_id in np.unique(y):
        plt.scatter(X[y == class_id, 0], X[y == class_id, 1], label=f"Class {class_id}", alpha=0.6)
    plt.title("Generated 2D Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)


# ============================================================================
# BASE MLP CLASS
# ============================================================================
class BaseMLP(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.activation_fn = activation_fn
    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_fn(self.fc2(x))
        x = self.fc3(x)
        return x
    
# ============================================================================
# TRAINING WITH TORCH.SIGMOID (ORIGINAL)
# ============================================================================

print("=" * 60)
print("TRAINING WITH TORCH.SIGMOID (ORIGINAL)")
print("=" * 60)

class SimpleMLP(BaseMLP):
    def __init__(self):
        super().__init__(lambda x: torch.sigmoid(x))

model = SimpleMLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Track original training
torch_sigmoid_loss_history = []

for epoch in range(EPOCHS):
    model.train()
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    torch_sigmoid_loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        pred = torch.argmax(logits, dim=1)
        acc = accuracy_score(y_train.numpy(), pred.numpy())
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")


model.eval()
with torch.no_grad():
    test_logits = model(X_test)
    test_pred = torch.argmax(test_logits, dim=1)
    test_acc = accuracy_score(y_test.numpy(), test_pred.numpy())
    print(f"Test Accuracy: {test_acc:.4f}")

torch_final_loss = torch_sigmoid_loss_history[-1]
torch_test_acc = test_acc

# ============================================================================
# ADD FAST_SIGMOID VERSION 
# ============================================================================
# Add fast_sigmoid import
from extension_cpp.ops import fast_sigmoid

print("\n" + "=" * 60)
print("TRAINING WITH FAST_SIGMOID (REPLACEMENT)")
print("=" * 60)

# Create fast_sigmoid version of the same model
class FastSigmoidMLP(BaseMLP):
    def __init__(self):
        super().__init__(lambda x: fast_sigmoid(x, -5, 5, 2000))

# Train fast_sigmoid model with identical setup
model_fast = FastSigmoidMLP()
criterion_fast = nn.CrossEntropyLoss()
optimizer_fast = optim.Adam(model_fast.parameters(), lr=0.01)

# Track fast_sigmoid training
fast_sigmoid_loss_history = []

for epoch in range(EPOCHS):
    model_fast.train()
    logits = model_fast(X_train)
    loss = criterion_fast(logits, y_train)

    optimizer_fast.zero_grad()
    loss.backward()
    optimizer_fast.step()
    
    fast_sigmoid_loss_history.append(loss.item())

    if (epoch + 1) % 10 == 0:
        pred = torch.argmax(logits, dim=1)
        acc = accuracy_score(y_train.numpy(), pred.numpy())
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

model_fast.eval()
with torch.no_grad():
    test_logits_fast = model_fast(X_test)
    test_pred_fast = torch.argmax(test_logits_fast, dim=1)
    test_acc_fast = accuracy_score(y_test.numpy(), test_pred_fast.numpy())
    print(f"Test Accuracy: {test_acc_fast:.4f}")

fast_final_loss = fast_sigmoid_loss_history[-1]
fast_test_acc = test_acc_fast

# ============================================================================
# CONVERGENCE COMPARISON
# ============================================================================

print("\n" + "=" * 60)
print("CONVERGENCE COMPARISON RESULTS")
print("=" * 60)

loss_diff = abs(torch_final_loss - fast_final_loss)
acc_diff = abs(torch_test_acc - fast_test_acc)
loss_diff_percent = loss_diff / torch_final_loss * 100

print(f"{'Metric':<20} {'torch.sigmoid':<15} {'fast_sigmoid':<15} {'Difference':<15}")
print("-" * 65)
print(f"{'Final Loss':<20} {torch_final_loss:<15.6f} {fast_final_loss:<15.6f} {loss_diff:<15.6f}")
print(f"{'Test Accuracy':<20} {torch_test_acc:<15.4f} {fast_test_acc:<15.4f} {acc_diff:<15.4f}")
print(f"{'Loss Diff %':<20} {'-':<15} {'-':<15} {loss_diff_percent:<15.2f}%")

# How to assess convergence?

# ============================================================================
# PLOT TORCH.SIGMOID AND FAST_SIGMOID
# ============================================================================
if PLOT:
    def plot_decision_boundary(model, X, y, scaler):
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid)
        grid_tensor = torch.tensor(grid_scaled, dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            logits = model(grid_tensor)
            pred = torch.argmax(logits, dim=1).numpy()

        Z = pred.reshape(xx.shape)
        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        for class_id in np.unique(y):
            plt.scatter(X[y == class_id, 0], X[y == class_id, 1], label=f"Class {class_id}", edgecolor='k')
        plt.title("Decision Boundary")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_decision_boundary(model, X, y, scaler)
    plot_decision_boundary(model_fast, X, y, scaler)


# ============================================================================
# PLOT LOSS CURVES
# ============================================================================

if PLOT:
    plt.figure(figsize=(8, 5))
    plt.plot(torch_sigmoid_loss_history, label='torch.sigmoid', linestyle='-', color='blue')
    plt.plot(fast_sigmoid_loss_history, label='fast_sigmoid', linestyle='--', color='orange')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()