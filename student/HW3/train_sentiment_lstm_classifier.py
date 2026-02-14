#!/usr/bin/env python
# coding: utf-8
"""
Task 2: LSTM with Padded FastText Word Vectors (32 tokens x 300 dim)
Financial PhraseBank (sentences_50agree) via Hugging Face Datasets.

Requirements satisfied:
- Tokenize sentence into word tokens, lookup FastText vectors (no nn.Embedding)
- Pad/truncate to exactly 32 tokens -> (32, 300) per sample
- Batch precomputed word vectors directly
- LSTM classifier using final hidden state
- Class-weighted CrossEntropyLoss
- Stratified splits: 15% test, then 15% of remaining for val (seed 42)
- Train at least 30 epochs; early stopping allowed only after epoch 30
- Track and plot train/val: loss, accuracy, macro F1
- Save plots + confusion matrix to disk in ./outputs
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

# ========== Constants ==========
MAX_TOKENS = 32
FASTTEXT_NAME = "fasttext-wiki-news-subwords-300"

# Tokenization (simple regex tokenizer)
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

def tokenize(text: str):
    return [t.lower() for t in _TOKEN_RE.findall(text)]

print("\n========== Loading Dataset ==========")
dataset = datasets.load_dataset("financial_phrasebank", "sentences_50agree")
print("Dataset loaded. Example:", dataset["train"][0])

print("\n========== Preparing DataFrame ==========")
data = pd.DataFrame(dataset["train"])
data["text_label"] = data["label"].apply(lambda x: "positive" if x == 2 else "neutral" if x == 1 else "negative")
print(f"DataFrame shape: {data.shape}")
print("Label distribution:\n", data["text_label"].value_counts())

# ========== Stratified Splits (required procedure) ==========
print("\n========== Splitting Data ==========")
y = data["label"].values
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
print(f"Loading: {FASTTEXT_NAME} (may download on first run)...")
kv = api.load(FASTTEXT_NAME)
EMB_DIM = kv.vector_size
print(f"FastText loaded. Embedding dim = {EMB_DIM}")

def sentence_to_padded_matrix(sentence: str) -> np.ndarray:
    """
    Returns (MAX_TOKENS, EMB_DIM) float32 matrix.
    - tokenizes
    - truncates/pads to MAX_TOKENS
    - OOV tokens become zero vectors (but still consume a position)
    """
    toks = tokenize(sentence)
    mat = np.zeros((MAX_TOKENS, EMB_DIM), dtype=np.float32)

    # Fill up to MAX_TOKENS positions
    for j in range(min(len(toks), MAX_TOKENS)):
        tok = toks[j]
        if tok in kv:
            mat[j] = kv[tok].astype(np.float32)
        # else leave zeros
    return mat

# ========== Precompute Sequences (N, 32, 300) ==========
print("\n========== Building (32, 300) Sequence Tensors ==========")
sentences = data["sentence"].tolist()
X_seq = np.zeros((len(sentences), MAX_TOKENS, EMB_DIM), dtype=np.float32)

for i, sent in enumerate(tqdm(sentences, desc="Vectorizing sentences")):
    X_seq[i] = sentence_to_padded_matrix(sent)

X_train_seq = X_seq[X_train]
X_val_seq = X_seq[X_val]
X_test_seq = X_seq[X_test]

y_train_arr = y[X_train]
y_val_arr = y[X_val]
y_test_arr = y[X_test]

print(f"X_train_seq: {X_train_seq.shape}, y_train: {y_train_arr.shape}")
print(f"X_val_seq:   {X_val_seq.shape}, y_val:   {y_val_arr.shape}")
print(f"X_test_seq:  {X_test_seq.shape}, y_test:  {y_test_arr.shape}")

# ========== PyTorch Dataset ==========
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 32, 300)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return {"x": self.X[idx], "labels": self.y[idx]}

train_dataset = SequenceDataset(X_train_seq, y_train_arr)
val_dataset = SequenceDataset(X_val_seq, y_val_arr)
test_dataset = SequenceDataset(X_test_seq, y_test_arr)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("DataLoaders created.")

# ========== Model Definition ==========
print("\n========== Defining LSTM Classifier ==========")
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.35, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,  # (B, T, D)
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, hidden_dim)

        if self.bidirectional:
            # last layer: forward = -2, backward = -1
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (B, 2*hidden_dim)
        else:
            h_last = h_n[-1]  # (B, hidden_dim)

        h_last = self.dropout(h_last)
        return self.fc(h_last)

num_classes = len(np.unique(y))
model = LSTMClassifier(
    input_dim=EMB_DIM,
    hidden_dim=256,
    num_layers=2,
    num_classes=num_classes,
    dropout=0.35,
    bidirectional=True
)

# ========== Training Setup ==========
print("\n========== Setting Up Training ==========")
device = get_device()
print(f"Using device: {device}")
os.makedirs("outputs", exist_ok=True)

model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

# Class weights computed from TRAIN split
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
MAX_EPOCHS = 60         # can be longer
MIN_EPOCHS = 30         # must train at least 30
PATIENCE = 6
patience_left = PATIENCE

best_val_f1 = 0.0

train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
train_acc_history = []
val_acc_history = []

for epoch in range(MAX_EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{MAX_EPOCHS} ---")

    # ---- Train ----
    model.train()
    running_loss = 0.0
    all_train_preds = []
    all_train_labels = []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
        xb = batch["x"].to(device)          # (B, 32, 300)
        labels = batch["labels"].to(device)

        logits = model(xb)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)

        all_train_preds.extend(preds.detach().cpu().numpy())
        all_train_labels.extend(labels.detach().cpu().numpy())

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_f1 = f1_score(all_train_labels, all_train_preds, average="macro")
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
    val_f1 = f1_score(all_val_labels, all_val_preds, average="macro")
    val_acc = (np.array(all_val_preds) == np.array(all_val_labels)).mean()

    val_loss_history.append(epoch_val_loss)
    val_f1_history.append(val_f1)
    val_acc_history.append(val_acc)

    print(f"Val Loss: {epoch_val_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

    scheduler.step(val_f1)

    # ---- Save best checkpoint ----
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "outputs/best_lstm_fasttext_model.pth")
        print(f">>> Saved new best model (Val F1: {best_val_f1:.4f})")
        patience_left = PATIENCE
    else:
        # Early stopping allowed only after MIN_EPOCHS
        if (epoch + 1) >= MIN_EPOCHS:
            patience_left -= 1
            if patience_left <= 0:
                print(f">>> Early stopping triggered after epoch {epoch+1} (no val F1 improvement for {PATIENCE} epochs).")
                break

# Save metric history (optional but helpful)
np.savez(
    "outputs/lstm_fasttext_history.npz",
    train_loss=np.array(train_loss_history),
    val_loss=np.array(val_loss_history),
    train_acc=np.array(train_acc_history),
    val_acc=np.array(val_acc_history),
    train_f1=np.array(train_f1_history),
    val_f1=np.array(val_f1_history),
)
print("Saved metric history to outputs/lstm_fasttext_history.npz")

# ========== Plot Learning Curves ==========
print("\n========== Plotting Learning Curves ==========")
plt.figure(figsize=(12, 15))

plt.subplot(3, 1, 1)
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(train_f1_history, label="Train F1")
plt.plot(val_f1_history, label="Val F1")
plt.title("Macro F1 Curve")
plt.xlabel("Epochs")
plt.ylabel("Macro F1")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(train_acc_history, label="Train Acc")
plt.plot(val_acc_history, label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("outputs/lstm_fasttext_learning_curves.png")
plt.show()
print("Learning curves saved as 'outputs/lstm_fasttext_learning_curves.png'.")

# ========== Test Evaluation ==========
print("\n========== Evaluating on Test Set ==========")
model.load_state_dict(torch.load("outputs/best_lstm_fasttext_model.pth", map_location=device))
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
test_f1_macro = f1_score(all_labels, all_preds, average="macro")
test_f1_weighted = f1_score(all_labels, all_preds, average="weighted")

print("\n" + "="*50)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Macro: {test_f1_macro:.4f}")
print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
print("="*50 + "\n")

class_names = ["Negative (0)", "Neutral (1)", "Positive (2)"]
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/lstm_fasttext_confusion_matrix.png")
plt.show()
print("Confusion matrix saved as 'outputs/lstm_fasttext_confusion_matrix.png'.")

print("\nPer-class F1 Scores:")
for i, name in enumerate(class_names):
    class_f1 = f1_score(all_labels, all_preds, labels=[i], average="macro")
    print(f"{name}: {class_f1:.4f}")

print("\n========== Script Complete ==========")
