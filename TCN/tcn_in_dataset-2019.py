"""
tcn_cross_dataset.py

Train a Temporal Convolutional Network (TCN) on one cleaned dataset
and evaluate it on another cleaned dataset, with training visuals.

Make sure you first ran clean_data_for_tcn.py and created *_tcn.csv files.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ========= CONFIG: set your cleaned file paths here =========
TRAIN_FILE = "CICIDS2019-tcn.csv"
TEST_FILE  = "CICIDS2019-tcn.csv"

BATCH_SIZE = 512
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3
DROPOUT = 0.3
NUM_CHANNELS = (64, 64, 128)   # TCN residual block channels
KERNEL_SIZE = 3
# ============================================================


def load_clean_dataset(path: str, scaler: StandardScaler = None, fit_scaler: bool = False):
    """
    Load cleaned CSV:
        - uses all columns except Label and Label_id as features
        - uses Label_id as target
        - applies StandardScaler (fit on train, transform on test)
    """
    df = pd.read_csv(path)

    if "Label_id" not in df.columns:
        raise ValueError(f"{path} must contain 'Label_id' column. "
                         f"Run clean_data_for_tcn.py first.")

    feature_cols = [c for c in df.columns if c not in ("Label", "Label_id")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Label_id"].values.astype(np.int64)

    if scaler is None and fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    return X_scaled, y, scaler, feature_cols


# ----------------- TCN implementation -----------------

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x):
        # x: (B, C, L)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # crop to original length
        out = out[:, :, :x.size(2)]

        res = x if self.downsample is None else self.downsample(x)
        res = res[:, :, :out.size(2)]

        return self.relu(out + res)


class TCNClassifier(nn.Module):
    def __init__(self, num_inputs, num_classes,
                 num_channels=(64, 64, 128),
                 kernel_size=3,
                 dropout=0.3):
        super().__init__()

        layers = []
        in_ch = num_inputs
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i    # 1, 2, 4, ...
            layers.append(
                TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch

        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x: (B, C, L). For tabular data, L = 1
        out = self.network(x)
        out = out.mean(dim=2)          # global avg pool over "time"
        logits = self.classifier(out)
        return logits


def train_and_eval_tcn(X_train, y_train, X_test, y_test,
                       num_channels=NUM_CHANNELS,
                       kernel_size=KERNEL_SIZE,
                       dropout=DROPOUT,
                       lr=LEARNING_RATE,
                       batch_size=BATCH_SIZE,
                       num_epochs=NUM_EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    num_features = X_train.shape[1]
    num_classes = 8


    # (N, F) -> (N, F, 1) for Conv1d
    X_train_t = torch.from_numpy(X_train).float().unsqueeze(-1)
    X_test_t = torch.from_numpy(X_test).float().unsqueeze(-1)
    y_train_t = torch.from_numpy(y_train).long()
    y_test_t = torch.from_numpy(y_test).long()

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    model = TCNClassifier(
        num_inputs=num_features,
        num_classes=num_classes,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --------- TRACKING FOR VISUALS ---------
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # ------------- training loop -------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total

        # ---- Evaluate on test set each epoch for curve ----
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct_test += (preds == yb).sum().item()
                total_test += yb.size(0)
        test_acc = correct_test / total_test

        # store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_acc:.4f} | "
            f"Test acc: {test_acc:.4f}"
        )

    # ------------- final detailed evaluation on test set -------------
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(yb.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    print("\n===== Cross-dataset evaluation (TCN) =====")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Test  file: {TEST_FILE}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 score : {f1:.4f}")

    # ------------- VISUALS -------------
    epochs = range(1, num_epochs + 1)

    # Accuracy curves
    plt.figure()
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("TCN Training vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Loss curve
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TCN Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model


def main():
    # Load train dataset + fit scaler
    X_train, y_train, scaler, feat_cols = load_clean_dataset(
        TRAIN_FILE, scaler=None, fit_scaler=True
    )
    
    SAMPLE_SIZE = 200_000  # 200k samples

    if len(X_train) > SAMPLE_SIZE:
        idx = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

   
    print(f"[INFO] Train set: {X_train.shape}, features: {len(feat_cols)}")

    # Load test dataset using same scaler (cross-dataset evaluation)
    X_test, y_test, _, _ = load_clean_dataset(
        TEST_FILE, scaler=scaler, fit_scaler=False
    )
    TEST_SIZE = 200_000

    if len(X_test) > TEST_SIZE:
        idx = np.random.choice(len(X_test), TEST_SIZE, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    print(f"[INFO] Test set : {X_test.shape}")

    _ = train_and_eval_tcn(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
