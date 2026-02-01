import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, pairs):
        pairs = torch.tensor(pairs, dtype=torch.long)
        self.centers = pairs[:, 0]
        self.contexts = pairs[:, 1]

    def __len__(self):
        return self.centers.size(0)

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)   # center words
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)  # context words

        nn.init.uniform_(self.in_embed.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_idx, context_idx):
        v = self.in_embed(center_idx)
        u = self.out_embed(context_idx)
        return (v * u).sum(dim=-1)   # dot product

    def get_embeddings(self):
        return self.in_embed.weight.detach().cpu().numpy()

# build unigram distribution
def uni_distr(word_counts, vocab_size, power=0.75):
    if isinstance(word_counts, (list, tuple)):
        counts = torch.tensor(word_counts, dtype=torch.float)
    elif isinstance(word_counts, torch.Tensor):
        counts = word_counts.float()
    elif isinstance(word_counts, dict):
        counts = torch.zeros(vocab_size, dtype=torch.float)
        for k, v in word_counts.items():
            if isinstance(k, int) and 0 <= k < vocab_size:
                counts[k] = float(v)
    else:
        raise TypeError("word_counts must be dict/list/tensor")

    counts = torch.clamp(counts, min=1.0)  # avoid zeros
    probs = counts.pow(power)
    probs = probs / probs.sum()
    return probs

@torch.no_grad()
def sample_negatives(neg_probs, batch_size, num_neg, device, forbidden=None):
    negs = torch.multinomial(neg_probs, batch_size * num_neg, replacement=True).view(batch_size, num_neg)

    if forbidden is None:
        return negs.to(device)

    forbidden = forbidden.view(-1, 1)
    mask = (negs == forbidden)
    while mask.any():
        # resample only masked positions
        n_bad = int(mask.sum().item())
        resampled = torch.multinomial(neg_probs, n_bad, replacement=True)
        negs[mask] = resampled
        mask = (negs == forbidden)

    return negs.to(device)


ROOT = "/home/sagemaker-user/"


def main():
    # Load data
    with open(ROOT + "stat359/student/HW2/processed_data.pkl", "rb") as f:
        data = pickle.load(f)

    word2idx = data["word2idx"]
    idx2word = data["idx2word"]
    vocab_size = len(word2idx)

    device = select_device()
    print(f"Using device: {device}")

    # Extract skip-gram pairs
    skipgram_df = data["skipgram_df"]

    possible_cols = [
        ("center", "context"),
        ("center_idx", "context_idx"),
        ("input", "output"),
        ("w_i", "w_o"),
    ]

    for c1, c2 in possible_cols:
        if c1 in skipgram_df.columns and c2 in skipgram_df.columns:
            pairs = skipgram_df[[c1, c2]].to_numpy()
            break
    else:
        raise KeyError(f"Could not find center/context columns in skipgram_df")

    # Negative sampling distribution
    counter = data["counter"]

    counts = torch.zeros(vocab_size, dtype=torch.float)
    for w, c in counter.items():
        if w in word2idx:
            counts[word2idx[w]] = float(c)

    neg_probs = uni_distr(counts, vocab_size, power=0.75).to(device)

    # Dataset and DataLoader
    dataset = SkipGramDataset(pairs)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model, Loss, Optimizer
    model = Word2Vec(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for center, context in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            center = center.to(device)
            context = context.to(device)
            B = center.size(0)

            # Positive
            pos_logits = model(center, context)
            pos_labels = torch.ones(B, device=device)

            # Negatives
            neg_context = sample_negatives(
                neg_probs=neg_probs,
                batch_size=B,
                num_neg=NEGATIVE_SAMPLES,
                device=device,
                forbidden=context
            )

            center_exp = center.unsqueeze(1).expand(-1, NEGATIVE_SAMPLES)  # (B, K)
            neg_logits = model(center_exp, neg_context)                    # (B, K)
            neg_labels = torch.zeros(B, NEGATIVE_SAMPLES, device=device)

            loss = criterion(pos_logits, pos_labels) + criterion(neg_logits, neg_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        print(f"Epoch {epoch}: avg loss = {total_loss / len(loader):.4f}")

    # Save embeddings and mappings
    embeddings = model.get_embeddings()
    with open("word2vec_embeddings.pkl", "wb") as f:
        pickle.dump({"embeddings": embeddings, "word2idx": word2idx, "idx2word": idx2word}, f)

    print("Embeddings saved to word2vec_embeddings.pkl")


# Device selection: CUDA > MPS > CPU
def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# call
if __name__ == "__main__":
    main()

