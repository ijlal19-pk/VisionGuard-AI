import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json

# Configuration
BASE_DIR = r"c:\Users\ijlal\Desktop\GLUCOMA DETECTOR"
DATA_DIR = os.path.join(BASE_DIR, "data", "test")
MODEL_PATH = os.path.join(BASE_DIR, "model", "glaucoma_model.pth")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
HISTORY_PATH = os.path.join(BASE_DIR, "model", "history.json")

# 1. Load Model
device = torch.device("cpu") # Run locally on CPU
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# 2. Prepare Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"Classes detected: {dataset.classes}")
print(f"Class-to-index mapping: {dataset.class_to_idx}")

# 3. Evaluation
all_preds = []
all_labels = []
all_probs = []

print("Running Evaluation on Test Set...")
with torch.no_grad():
    for inputs, labels in loader:
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.numpy()[:, 1])

# 4. Metrics
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=dataset.classes)
auc = roc_auc_score(all_labels, all_probs)

# Calculate sensitivity and specificity
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)

print("\n--- PERFORMANCE REPORT ---")
print(report)
print(f"AUC-ROC Score: {auc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Save metrics to JSON for use in Streamlit app
metrics = {
    "accuracy": round(accuracy * 100, 2),
    "sensitivity": round(sensitivity * 100, 2),
    "specificity": round(specificity * 100, 2),
    "auc_roc": round(auc, 4),
    "total_test_images": len(all_labels),
    "true_positives": int(tp),
    "true_negatives": int(tn),
    "false_positives": int(fp),
    "false_negatives": int(fn)
}

# 5. Visualizations
os.makedirs(RESULTS_DIR, exist_ok=True)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.close()

# Plot ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150)
plt.close()

# Plot Training History (if history.json exists)
if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH, 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=150)
    plt.close()

# Save metrics JSON
with open(os.path.join(RESULTS_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\nEvaluation Complete! Results saved to: {RESULTS_DIR}")
