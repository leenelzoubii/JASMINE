"""
Deep Learning models: LSTM and Transformer classifiers.
Built with PyTorch for sequence-based classification.
"""
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SequenceDataset(Dataset):
    """Dataset for variable-length keypoint sequences."""
    def __init__(self, sequences: List[np.ndarray], labels: np.ndarray):
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences])
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
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMClassifier(nn.Module):
    """LSTM-based sequence classifier with configurable architecture."""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, (hidden, _) = self.lstm(packed)
        else:
            lstm_out, (hidden, _) = self.lstm(x)
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        hidden_cat = self.layer_norm(hidden_cat)
        out = self.dropout(hidden_cat)
        logits = self.fc(out)
        return logits


class TransformerClassifier(nn.Module):
    """Transformer-based sequence classifier with configurable architecture."""
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 4, num_classes: int = 2, dropout: float = 0.2,
                 max_seq_len: int = 500):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        src_key_padding_mask = None
        if lengths is not None:
            max_len = x.size(1)
            src_key_padding_mask = torch.arange(max_len, device=x.device)[None, :] >= lengths[:, None]
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        output = self.layer_norm(output)
        if src_key_padding_mask is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)
            output = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            output = output.mean(dim=1)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits


class DLModelTrainer:
    """Trainer for deep learning models with hyperparameter options."""
    def __init__(self, model_type: str = 'lstm', input_size: int = 75,
                 num_classes: int = 2, device: Optional[str] = None,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, d_model: int = 128,
                 nhead: int = 8, transformer_layers: int = 4):
        self.model_type = model_type
        self.input_size = input_size
        self.num_classes = num_classes
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f"  Using device: {self.device}")

        if model_type == 'lstm':
            self.model = LSTMClassifier(
                input_size=input_size, hidden_size=hidden_size,
                num_layers=num_layers, num_classes=num_classes, dropout=dropout,
            ).to(self.device)
        elif model_type == 'transformer':
            self.model = TransformerClassifier(
                input_size=input_size, d_model=d_model, nhead=nhead,
                num_layers=transformer_layers, num_classes=num_classes,
                dropout=dropout,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.is_fitted = False

    def train(self, X_train: List[np.ndarray], y_train: np.ndarray,
              X_val: Optional[List[np.ndarray]] = None, y_val: Optional[List[np.ndarray]] = None,
              epochs: int = 100, batch_size: int = 32, lr: float = 0.001,
              weight_decay: float = 1e-4, patience: int = 15) -> Dict:
        train_dataset = SequenceDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = SequenceDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

        class_weights = self._compute_class_weights(y_train)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Warmup + cosine annealing scheduler
        warmup_epochs = min(10, epochs // 10)
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_model_state = None
        early_stop_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for sequences, labels, lengths in train_loader:
                sequences, labels = sequences.to(self.device), labels.to(self.device)
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
            current_lr = optimizer.param_groups[0]['lr']
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(avg_train_acc)
            history['lr'].append(current_lr)

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for sequences, labels, lengths in val_loader:
                        sequences, labels = sequences.to(self.device), labels.to(self.device)
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

                scheduler.step()

                # Early stopping + best model tracking
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = avg_val_acc
                    best_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if (epoch + 1) % 10 == 0 or epoch == 0 or early_stop_counter == patience // 2:
                    print(f"  Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f} | LR: {current_lr:.2e}")

                if early_stop_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                scheduler.step()

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"  Restored best model (val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.4f})")

        self.is_fitted = True

        metrics = {
            'model_type': self.model_type,
            'final_train_loss': avg_train_loss,
            'final_train_acc': avg_train_acc,
            'epochs_trained': epoch + 1,
        }
        if val_loader is not None:
            metrics['best_val_loss'] = best_val_loss
            metrics['best_val_acc'] = best_val_acc
        return metrics

    def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor:
        """Compute inverse frequency class weights."""
        counts = np.bincount(y)
        if len(counts) < 2:
            return torch.ones(2, device=self.device)
        weights = 1.0 / counts.astype(np.float32)
        weights /= weights.sum()
        weights = torch.tensor(weights * len(counts), dtype=torch.float32, device=self.device)
        return weights

    def predict(self, X: List[np.ndarray]) -> np.ndarray:
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
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        state = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'input_size': self.input_size,
            'num_classes': self.num_classes,
        }
        if self.model_type == 'lstm':
            state['hidden_size'] = self.model.hidden_size
            state['num_layers'] = self.model.num_layers
        elif self.model_type == 'transformer':
            state['d_model'] = self.model.input_proj.out_features
            state['nhead'] = self.model.transformer_encoder.layers[0].self_attn.num_heads
            state['num_layers'] = len(self.model.transformer_encoder.layers)
            state['dropout'] = self.model.dropout.p
        torch.save(state, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        model_type = checkpoint['model_type']
        input_size = checkpoint['input_size']
        num_classes = checkpoint['num_classes']
        self.model_type = model_type
        self.input_size = input_size
        self.num_classes = num_classes

        if model_type == 'lstm':
            hs = checkpoint.get('hidden_size', 128)
            nl = checkpoint.get('num_layers', 2)
            dp = checkpoint.get('dropout', 0.3)
            self.model = LSTMClassifier(input_size, hs, nl, num_classes, dp).to(self.device)
        elif model_type == 'transformer':
            dm = checkpoint.get('d_model', 128)
            nh = checkpoint.get('nhead', 8)
            nl = checkpoint.get('num_layers', 4)
            dp = checkpoint.get('dropout', 0.2)
            self.model = TransformerClassifier(input_size, dm, nh, nl, num_classes, dp).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.is_fitted = True
