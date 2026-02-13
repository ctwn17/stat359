"""
Script for training an MLP-based sentiment classifier on the financial_phrasebank dataset
using mean-pooled FastText (fasttext-wiki-news-subwords-300) sentence embeddings.
"""

# ========== Imports ==========
import os
import re
import random
import numpy as np
import pandas as pd
import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gensim.downloader as api

# ========== Reproducibility ==========
SEED = 5211314
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


print("\n========== Loading Dataset ==========")
dataset = datasets.load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print("Dataset loaded. Example:", dataset['train'][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset['train'])
data['text_label'] = data['label'].apply(lambda x: 'positive' if x == 2 else 'neutral' if x == 1 else 'negative')
print(f"DataFrame shape: {data.shape}")
print("Label distribution:\n", data['text_label'].value_counts())

# ========== Train/Test Split (required stratified procedure) ==========
print("\n========== Splitting Data ==========")
y = data['label'].values
idx = np.arange(len(y))

X_trainval, X_test, y_trainval, y_test = train_test_split(
    idx, y, test_size=0.15, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=SEED
)
print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ========== Load FastText ==========
print("\n========== Loading FastText Embeddings ==========")
print("Loading: fasttext-wiki-news-subwords-300 (may download on first run)...")
kv = api.load("fasttext-wiki-news-subwords-300")
EMB_DIM = kv.vector_size
print(f"FastText loaded. Embedding dim = {EMB_DIM}")

# ========== Tokenization + Mean Pooling ==========
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

def tokenize(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(text)]

def mean_pool_sentence(text: str):
    toks = tokenize(text)
    vecs = []
    for t in toks:
        if t in kv:
            vecs.append(kv[t])
    if not vecs:
        return np.zeros(EMB_DIM, dtype=np.float32)
    return np.mean(np.asarray(vecs, dtype=np.float32), axis=0)

print("\n========== Building Sentence Embeddings ==========")
X_all = np.zeros((len(data), EMB_DIM), dtype=np.float32)
for i, sent in enumerate(tqdm(data['sentence'].tolist(), desc="Embedding sentences")):
    X_all[i] = mean_pool_sentence(sent)

X_train_emb = X_all[X_train]
X_val_emb = X_all[X_val]
X_test_emb = X_all[X_test]

y_train_arr = y[X_train]
y_val_arr = y[X_val]
y_test_arr = y[X_test]

print(f"X_train_emb: {X_train_emb.shape}, y_train: {y_train_arr.shape}")
print(f"X_val_emb:   {X_val_emb.shape}, y_val:   {y_val_arr.shape}")
print(f"X_test_emb:  {X_test_emb.shape}, y_test:  {y_test_arr.shape}")

# ========== PyTorch Dataset ==========
class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {
            "x": self.X[idx],
            "labels": self.y[idx]
        }

train_dataset = EmbeddingDataset(X_train_emb, y_train_arr)
val_dataset = EmbeddingDataset(X_val_emb, y_val_arr)
test_dataset = EmbeddingDataset(X_test_emb, y_test_arr)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("DataLoaders created.")

# ========== Model Definition ==========
print("\n========== Defining MLP Classifier ==========")
class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)

num_classes = len(np.unique(y))
model = MLPClassifier(EMB_DIM, hidden_dim=256, num_classes=num_classes, dropout=0.25)

# ========== Training Setup ==========
print("\n========== Setting Up Training ==========")
device = get_device()
print(f"Using device: {device}")
os.makedirs("outputs", exist_ok=True)
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Class weights computed from TRAIN split (recommended + reproducible)
counts = np.bincount(y_train_arr, minlength=num_classes).astype(np.float64)
counts[counts == 0] = 1.0
class_weights = 1.0 / torch.tensor(counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum()
class_weights = class_weights.to(device)

print(f"Train class counts: {counts.tolist()}")
print(f"Class weights: {class_weights.detach().cpu().numpy()}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
print("Training setup complete.")

# ========== Training Loop ==========
print("\n========== Starting Training Loop ==========")
num_epochs = 30
best_val_f1 = 0.0

train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(num_epochs):
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

    # ---- Train ----
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        xb = batch["x"].to(device)
        labels = batch["labels"].to(device)

        logits = model(xb)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)

        all_train_preds.extend(preds.detach().cpu().numpy())
        all_train_labels.extend(labels.detach().cpu().numpy())

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_train_labels, all_train_preds, average='macro')
    train_acc = (np.array(all_train_preds) == np.array(all_train_labels)).mean()

    train_loss_history.append(epoch_train_loss)
    train_f1_history.append(train_f1)
    train_acc_history.append(train_acc)

    print(f"Train Loss: {epoch_train_loss:.4f}, Train F1: {train_f1:.4f}, Train Acc: {train_acc:.4f}")

    # ---- Validation ----
    model.eval()
    val_loss = 0.0
    all_val_preds = []
    all_val_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=False):
            xb = batch["x"].to(device)
            labels = batch["labels"].to(device)

            logits = model(xb)
            loss = criterion(logits, labels)

            val_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)

            all_val_preds.extend(preds.detach().cpu().numpy())
            all_val_labels.extend(labels.detach().cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_acc = (np.array(all_val_preds) == np.array(all_val_labels)).mean()

    val_loss_history.append(epoch_val_loss)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)

    print(f"Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'outputs/best_mlp_fasttext_model.pth')
        print(f'>>> Saved new best model (Val F1: {best_val_f1:.4f})')

# ========== Plot Learning Curves ==========
print("\n========== Plotting Learning Curves ==========")
plt.figure(figsize=(12, 15))

plt.subplot(3, 1, 1)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(train_f1_history, label='Train F1')
plt.plot(val_f1_history, label='Val F1')
plt.title('F1 Macro Score Curve')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(val_acc_history, label='Val Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('outputs/mlp_fasttext_learning_curves.png')
plt.show()
print("Learning curves saved as 'outputs/mlp_fasttext_learning_curves.png'.")

# ========== Test Evaluation ==========
print("\n========== Evaluating on Test Set ==========")
model.load_state_dict(torch.load('outputs/best_mlp_fasttext_model.pth', map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        xb = batch["x"].to(device)
        labels = batch["labels"].to(device)

        logits = model(xb)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')

print('\n' + '='*50)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1_macro:.4f}")
print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
print('='*50 + '\n')

class_names = ['Negative (0)', 'Neutral (1)', 'Positive (2)']
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.savefig('outputs/mlp_fasttext_confusion_matrix.png')
plt.show()
print("Confusion matrix saved as 'outputs/mlp_fasttext_confusion_matrix.png'.")

print("\nPer-class F1 Scores:")
for i, name in enumerate(class_names):
    class_f1 = f1_score(all_labels, all_preds, labels=[i], average='macro')
    print(f"{name}: {class_f1:.4f}")

print("\n========== Script Complete ==========")
# End of script
