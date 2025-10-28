import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, depth=3, dropout=0.12):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, hidden),
                )
            )
        self.out_bn = nn.LayerNorm(hidden)

    def forward(self, x):
        x = F.relu(self.input(x))
        for b in self.blocks:
            res = b(x)
            x = F.relu(x + res)
        return self.out_bn(x)


class MDNHead(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures=6):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.out_dim = out_dim
        self.pi = nn.Linear(in_dim, n_mixtures)
        self.mu = nn.Linear(in_dim, n_mixtures * out_dim)
        self.logsigma = nn.Linear(in_dim, n_mixtures * out_dim)

    def forward(self, h):
        pi_log = F.log_softmax(self.pi(h), dim=-1)
        mu = self.mu(h).view(h.shape[0], self.n_mixtures, self.out_dim)
        logsigma = self.logsigma(h).view(h.shape[0], self.n_mixtures, self.out_dim)
        sigma = torch.exp(logsigma).clamp(min=1e-6)
        return pi_log, mu, sigma


class ProbModel(nn.Module):
    def __init__(self, in_dim, target_dim, latent=256, n_mixtures=6, dropout=0.12):
        super().__init__()
        self.encoder = ResidualMLP(in_dim, hidden=latent, depth=3, dropout=dropout)
        self.mdn = MDNHead(latent, target_dim, n_mixtures=n_mixtures)

    def forward(self, x):
        h = self.encoder(x)
        return self.mdn(h)


def mdn_nll(pi_log, mu, sigma, y):
    B, K, D = mu.shape
    y_exp = y.unsqueeze(1).expand(-1, K, -1)
    var = sigma * sigma + 1e-9
    log_norm = (
        -0.5 * (((y_exp - mu) ** 2) / var).sum(dim=2)
        - 0.5 * D * math.log(2 * math.pi)
        - 0.5 * torch.log(var).sum(dim=2)
    )
    log_comp = pi_log + log_norm
    log_prob = torch.logsumexp(log_comp, dim=1)
    loss = -log_prob.mean()
    return loss


def load_and_prepare(path):
    df = pd.read_csv(path)
    features = ["ra", "dec", "teff", "logg", "fe_h", "snr", "parallax"]
    target = ["z"]
    for col in features + target:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col] = 0.0
    X = df[features].values.astype(np.float32)
    y = df[target].values.astype(np.float32)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y)
    return X, y, scaler_x, scaler_y


if __name__ == "__main__":
    DATA_CSV = os.path.join("datasets", "ml_preprocessed", "all_sources.csv")
    X, y, scaler_x, scaler_y = load_and_prepare(DATA_CSV)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.12, random_state=42
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).squeeze(1)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).squeeze(1)
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    batch_size = 1024
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = X_train.shape[1]
    model = ProbModel(
        in_dim=in_dim, target_dim=1, latent=256, n_mixtures=6, dropout=0.12
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    epochs = 40
    best_val = 1e9
    out_dir = os.path.join("datasets", "ml_preprocessed", "models")
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            pi_log, mu, sigma = model(xb)
            loss = mdn_nll(pi_log, mu, sigma, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device).unsqueeze(1)
                pi_log, mu, sigma = model(xb)
                loss = mdn_nll(pi_log, mu, sigma, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)
        torch.save(model.state_dict(), os.path.join(out_dir, f"mdn_epoch{epoch}.pt"))
        print(f"epoch={epoch} train={train_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "mdn_best.pt"))
    torch.save(scaler_x, os.path.join(out_dir, "scaler_x.pth"))
    torch.save(scaler_y, os.path.join(out_dir, "scaler_y.pth"))
