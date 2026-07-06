import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class HIVDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[np.ndarray] = None,
                 tokenizer: RobertaTokenizer, max_len: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class RoBERTaMultiLabel(nn.Module):
    def __init__(self, n_labels: int = 8, dropout: float = 0.1):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def train_classifier(
    train_texts: List[str],
    train_labels: np.ndarray,
    val_texts: List[str],
    val_labels: np.ndarray,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    patience: int = 3
) -> Tuple[RoBERTaMultiLabel, Dict]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RoBERTaMultiLabel(n_labels=8, dropout=0.1).to(device)

    # Compute inverse frequency weights
    pos_counts = train_labels.sum(axis=0)
    total = len(train_labels)
    class_weights = total / (8 * pos_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    train_dataset = HIVDataset(train_texts, train_labels, tokenizer)
    val_dataset = HIVDataset(val_texts, val_labels, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    best_val_f1 = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        history['val_macro_f1'].append(macro_f1)

        logger.info(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_macro_f1={macro_f1:.4f}")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(f"{output_dir}/best_model.pt"))
    return model, history


def optimize_thresholds(model, val_loader, device):
    """Grid search per-class thresholds on validation set."""
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)

    best_thresholds = []
    for i in range(8):
        best_f1 = 0.0
        best_t = 0.5
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs[:, i] >= t).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds.append(best_t)
    return best_thresholds
