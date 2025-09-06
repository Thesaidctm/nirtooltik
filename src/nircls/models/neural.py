from __future__ import annotations

import numpy as np
from typing import Optional
from sklearn.base import BaseEstimator, ClassifierMixin

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None


class _BaseTorchClassifier(BaseEstimator, ClassifierMixin):
    """Base class for torch-based estimators."""

    def __init__(
        self,
        n_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        random_state: Optional[int] = None,
        device: str = "cpu",
        patience: int = 0,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.random_state = random_state
        self.device = device
        self.patience = patience

    def _build_model(self, n_features: int, n_classes: int) -> "nn.Module":  # pragma: no cover - abstract
        raise NotImplementedError

    # scikit-learn API -------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        if torch is None:
            raise ImportError(
                "torch is required for neural models. Install with `pip install torch`.")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        classes, y_indices = np.unique(y, return_inverse=True)
        self.classes_ = classes
        n_features = X.shape[1]
        n_classes = len(classes)

        device = torch.device(self.device)
        self.model_ = self._build_model(n_features, n_classes).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_indices))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val)
            class_to_idx = {c: i for i, c in enumerate(classes)}
            y_val_indices = np.array([class_to_idx[yy] for yy in y_val])
            val_ds = TensorDataset(
                torch.from_numpy(X_val), torch.from_numpy(y_val_indices)
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        best_state = None
        best_loss = float("inf")
        epochs_no_improve = 0

        for _ in range(self.n_epochs):
            self.model_.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        logits = self.model_(xb)
                        loss = criterion(logits, yb)
                        val_loss += loss.item() * xb.size(0)
                val_loss /= len(val_loader.dataset)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if self.patience and epochs_no_improve >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.device_ = device
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if torch is None:
            raise ImportError(
                "torch is required for neural models. Install with `pip install torch`.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X).to(self.device_))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


class SpectralCNN(_BaseTorchClassifier):
    """Simple 1D CNN for spectral classification."""

    def __init__(
        self,
        n_filters: int = 32,
        kernel_size: int = 3,
        hidden_dim: int = 64,
        n_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        random_state: Optional[int] = None,
        device: str = "cpu",
        patience: int = 0,
    ) -> None:
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            random_state=random_state,
            device=device,
            patience=patience,
        )
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

    def _build_model(self, n_features: int, n_classes: int) -> "nn.Module":
        if torch is None:  # pragma: no cover - handled in fit
            raise ImportError
        model = nn.Sequential(
            nn.Unflatten(1, (1, n_features)),
            nn.Conv1d(1, self.n_filters, self.kernel_size),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.n_filters, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, n_classes),
        )
        return model


class SpectralLSTM(_BaseTorchClassifier):
    """Simple LSTM-based spectral classifier."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 1,
        n_epochs: int = 10,
        batch_size: int = 32,
        lr: float = 1e-3,
        random_state: Optional[int] = None,
        device: str = "cpu",
        patience: int = 0,
    ) -> None:
        super().__init__(
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            random_state=random_state,
            device=device,
            patience=patience,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _build_model(self, n_features: int, n_classes: int) -> "nn.Module":
        if torch is None:  # pragma: no cover
            raise ImportError

        class Net(nn.Module):
            def __init__(self, n_features: int, n_classes: int, hidden_dim: int, num_layers: int):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_dim, n_classes)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.unsqueeze(-1)  # (batch, seq_len, 1)
                out, _ = self.lstm(x)
                out = out[:, -1, :]
                return self.fc(out)

        return Net(n_features, n_classes, self.hidden_dim, self.num_layers)
