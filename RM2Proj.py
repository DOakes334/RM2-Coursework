import os
import urllib.request
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --- DATA ACQUISITION ---
data_url = "https://raw.githubusercontent.com/cbannard/lela60331_24-25/refs/heads/main/coursework/Compiled_Reviews.txt"
data_file = "Compiled_Reviews.txt"

if not os.path.exists(data_file):
    print(f"Downloading data from {data_url}...")
    urllib.request.urlretrieve(data_url, data_file)
    print("Download complete.")

# --- DATA LOADING ---
reviews = []
sentiment_ratings = []

with open(data_file, "r", encoding="utf-8") as f:
   for line in f.readlines()[1:]:
        fields = line.rstrip().split('\t')
        if len(fields) >= 4:
            reviews.append(fields[0])
            sentiment_str = fields[1].strip().lower()
            sentiment_val = 1 if sentiment_str == 'positive' else 0
            sentiment_ratings.append(sentiment_val)

# --- PREPROCESSING ---
y = np.array(sentiment_ratings)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2), min_df=3)
X = vectorizer.fit_transform(reviews).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32).to(device)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1).to(device)

num_positives = sum(y_train)
num_negatives = len(y_train) - num_positives
pos_weight_val = num_negatives / num_positives
pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
print(f"Calculated pos_weight for class imbalance: {pos_weight_val:.4f}")

# --- MODEL DEFINITIONS ---

class Model1(nn.Module):
    def __init__(self, input_dim):
        super(Model1, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

class Model2(nn.Module):
    def __init__(self, input_dim):
        super(Model2, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), 

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), 

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3), 

            nn.Linear(64, 1)
          
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)

# --- TRAINING PIPELINES ---

# Baseline
def run_pipeline(model, X_train, y_train, X_test):
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5) 
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(X_test).cpu().numpy()

# Model 2
def run_pipeline_m2(model, X_train, y_train, X_test, pos_weight, batch_size=64, epochs=500, patience=25):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    val_size = int(0.1 * len(X_train))
    X_tr, X_val = X_train[:-val_size], X_train[-val_size:]
    y_tr, y_val = y_train[:-val_size], y_train[-val_size:]

    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_auc = 0.0
    best_weights = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val)).cpu().numpy()
        
        val_auc = roc_auc_score(y_val.cpu().numpy(), val_probs)
        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best val AUC: {best_auc:.4f})")
                break

    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_test)).cpu().numpy()

# --- RUN MODELS ---
print("Training Model 1 (Baseline)...")
m1_probs = run_pipeline(Model1(X_train.shape[1]).to(device), X_train_t, y_train_t, X_test_t)

print("Training Model 2 (Improved)...")
m2_probs = run_pipeline_m2(Model2(X_train.shape[1]).to(device), X_train_t, y_train_t, X_test_t, pos_weight)

# --- RESULTS ---
auc1 = roc_auc_score(y_test, m1_probs)
auc2 = roc_auc_score(y_test, m2_probs)

print(f"\nModel 1 AUC: {auc1:.4f}")
print(f"Model 2 AUC: {auc2:.4f}")
print(f"Improvement: +{auc2 - auc1:.4f}")

# --- BOOTSTRAPPING ---
def bootstrap_p_value(y_true, p1, p2, iterations=1000):
    observed_diff = roc_auc_score(y_true, p2) - roc_auc_score(y_true, p1)
    
    count = 0
    indices = np.arange(len(y_true))

    for _ in range(iterations):
        b_idx = np.random.choice(indices, size=len(indices), replace=True)
        if len(np.unique(y_true[b_idx])) < 2:
            continue

        b_auc1 = roc_auc_score(y_true[b_idx], p1[b_idx])
        b_auc2 = roc_auc_score(y_true[b_idx], p2[b_idx])
        b_diff = b_auc2 - b_auc1

        if b_diff <= 0:
            count += 1

    return count / iterations

p_val = bootstrap_p_value(y_test, m1_probs, m2_probs)

if p_val == 0.0:
    print("Bootstrapped p-value: < 0.001")
else:
    print(f"Bootstrapped p-value: {p_val:.4f}")