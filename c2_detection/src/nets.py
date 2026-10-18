"""PyTorch models (MLP + LSTM) and ONE shared training loop,
so both networks are trained in exactly the same way (fair comparison)."""
import copy
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Single-flow baseline. Input: (batch, n_features)."""

    def __init__(self, n_features: int, n_classes: int, hidden=(256, 128), dropout=0.3):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMClassifier(nn.Module):
    """Sequence model. Input: (batch, window, n_features).
    Reads the 10 flows in order; the hidden state after the LAST flow is classified."""

    def __init__(self, n_features: int, n_classes: int, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        out, _ = self.lstm(x)              # out: (batch, window, hidden)
        return self.head(out[:, -1, :])    # last time step -> 19 class scores


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict_proba(model, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Class probabilities (n, 19) for a numpy array of inputs."""
    dev = next(model.parameters()).device
    model.eval()
    outs = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).float().to(dev)
        outs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(outs)


def train_model(model, x_tr, y_tr, x_val, y_val, epochs, batch, lr, patience):
    """Adam + cross-entropy. After every epoch: weighted-F1 on validation.
    Keeps the best epoch; stops early if val-F1 has not improved for `patience` epochs."""
    dev = get_device()
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr).float(), torch.from_numpy(y_tr).long()),
                        batch_size=batch, shuffle=True)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        val_f1 = f1_score(y_val, predict_proba(model, x_val).argmax(1), average="weighted")
        history.append({"epoch": ep, "loss": total / len(x_tr), "val_f1": val_f1})
        print(f"  epoch {ep:>2}  loss={total / len(x_tr):.4f}  val_wF1={val_f1:.4f}  ({time.time() - t0:.0f}s)")

        if val_f1 > best_f1:
            best_f1, best_state, bad = val_f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop (best val_wF1={best_f1:.4f})")
                break
    model.load_state_dict(best_state)
    return model, history