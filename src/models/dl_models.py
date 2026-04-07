"""
Deep Learning models: LSTM and Transformer classifiers.

Built with PyTorch for sequence-based classification.
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """Dataset for variable-length keypoint sequences."""

    def __init__(self, sequences: List[np.ndarray], labels: np.ndarray):
        """
        Args:
            sequences: List of numpy arrays, each shape (seq_len, n_features)
            labels: numpy array of shape (n_samples,)
        """
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    """Collate function for variable-length sequences with padding."""
    sequences, labels = zip(*batch)

    # Get lengths
    lengths = torch.tensor([len(s) for s in sequences])

    # Pad sequences
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    labels = torch.stack(labels)

    return sequences_padded, labels, lengths


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMClassifier(nn.Module):
    """LSTM-based sequence classifier."""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3):
        """
        Args:
            input_size: Number of input features per timestep
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional

    def forward(self, x, lengths=None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
            lengths: Tensor of sequence lengths (optional, for packing)

        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        # Pack sequence if lengths provided
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, _) = self.lstm(packed)
        else:
            lstm_out, (hidden, _) = self.lstm(x)

        # Concatenate final hidden states from both directions
        hidden_fwd = hidden[-2]  # Last forward layer
        hidden_bwd = hidden[-1]  # Last backward layer
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)

        out = self.dropout(hidden_cat)
        logits = self.fc(out)
        return logits


class TransformerClassifier(nn.Module):
    """Transformer-based sequence classifier."""

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.1,
                 max_seq_len: int = 500):
        """
        Args:
            input_size: Number of input features per timestep
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            num_classes: Number of output classes
            dropout: Dropout probability
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths=None):
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_size)
            lengths: Tensor of sequence lengths (optional, for masking)

        Returns:
            logits: Tensor of shape (batch, num_classes)
        """
        # Project to d_model
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create mask for padding if lengths provided
        src_key_padding_mask = None
        if lengths is not None:
            max_len = x.size(1)
            src_key_padding_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]

        # Transformer encoder
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Global average pooling (only over non-padded positions)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            output = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            output = output.mean(dim=1)

        output = self.dropout(output)
        logits = self.fc(output)
        return logits


class DLModelTrainer:
    """Trainer for deep learning models."""

    def __init__(self, model_type: str = 'lstm', input_size: int = 75,
                 num_classes: int = 2, device: Optional[str] = None):
        """
        Args:
            model_type: 'lstm' or 'transformer'
            input_size: Number of input features per timestep
            num_classes: Number of output classes
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        self.model_type = model_type
        self.input_size = input_size
        self.num_classes = num_classes

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize model
        if model_type == 'lstm':
            self.model = LSTMClassifier(
                input_size=input_size,
                hidden_size=64,
                num_layers=2,
                num_classes=num_classes,
                dropout=0.3,
            ).to(self.device)
        elif model_type == 'transformer':
            self.model = TransformerClassifier(
                input_size=input_size,
                d_model=64,
                nhead=4,
                num_layers=2,
                num_classes=num_classes,
                dropout=0.1,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'lstm' or 'transformer'.")

        self.is_fitted = False

    def train(self, X_train: List[np.ndarray], y_train: np.ndarray,
              X_val: Optional[List[np.ndarray]] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> Dict:
        """
        Train the model.

        Args:
            X_train: List of training sequences, each shape (seq_len, n_features)
            y_train: Training labels
            X_val: Validation sequences (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            history: Dict with training history
        """
        # Create datasets and dataloaders
        train_dataset = SequenceDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = SequenceDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0
            )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        best_val_loss = float('inf')
        best_model_state = None

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for sequences, labels, lengths in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                logits = self.model(sequences, lengths)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item() * len(labels)
                _, predicted = torch.max(logits, 1)
                train_total += len(labels)
                train_correct += (predicted == labels).sum().item()

            avg_train_loss = train_loss / max(train_total, 1)
            avg_train_acc = train_correct / max(train_total, 1)

            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc)

            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for sequences, labels, lengths in val_loader:
                        sequences = sequences.to(self.device)
                        labels = labels.to(self.device)

                        logits = self.model(sequences, lengths)
                        loss = criterion(logits, labels)

                        val_loss += loss.item() * len(labels)
                        _, predicted = torch.max(logits, 1)
                        val_total += len(labels)
                        val_correct += (predicted == labels).sum().item()

                avg_val_loss = val_loss / max(val_total, 1)
                avg_val_acc = val_correct / max(val_total, 1)

                history['val_loss'].append(avg_val_loss)
                history['val_acc'].append(avg_val_acc)

                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                val_info = ""
                if val_loader is not None:
                    val_info = f" | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}"
                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
                      f"Train Acc: {avg_train_acc:.4f}{val_info}")

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted = True

        # Final metrics
        metrics = {
            'model_type': self.model_type,
            'final_train_loss': avg_train_loss,
            'final_train_acc': avg_train_acc,
        }
        if val_loader is not None:
            metrics['best_val_loss'] = best_val_loss
            metrics['best_val_acc'] = history['val_acc'][history['val_loss'].index(best_val_loss)]

        return metrics

    def predict(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: List of sequences, each shape (seq_len, n_features)

        Returns:
            predictions: numpy array of class labels
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        self.model.eval()
        dataset = SequenceDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        all_preds = []
        with torch.no_grad():
            for sequences, _, lengths in loader:
                sequences = sequences.to(self.device)
                logits = self.model(sequences, lengths)
                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())

        return np.array(all_preds)

    def predict_proba(self, X: List[np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: List of sequences

        Returns:
            probabilities: numpy array of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first.")

        self.model.eval()
        dataset = SequenceDataset(X, np.zeros(len(X)))
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        all_probs = []
        with torch.no_grad():
            for sequences, _, lengths in loader:
                sequences = sequences.to(self.device)
                logits = self.model(sequences, lengths)
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_probs)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
        }, path)

    def load(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)

        # Re-initialize model with correct input_size
        if checkpoint['model_type'] == 'lstm':
            self.model = LSTMClassifier(
                input_size=checkpoint['input_size'],
                hidden_size=64,
                num_layers=2,
                num_classes=checkpoint['num_classes'],
                dropout=0.3,
            ).to(self.device)
        elif checkpoint['model_type'] == 'transformer':
            self.model = TransformerClassifier(
                input_size=checkpoint['input_size'],
                d_model=64,
                nhead=4,
                num_layers=2,
                num_classes=checkpoint['num_classes'],
                dropout=0.1,
            ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_type = checkpoint['model_type']
        self.input_size = checkpoint['input_size']
        self.num_classes = checkpoint['num_classes']
        self.is_fitted = True
