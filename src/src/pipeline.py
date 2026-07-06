import os
import pandas as pd
from src.preprocessing.preprocess import full_preprocess_pipeline
from src.modeling.topic_modeling import train_bertopic, get_topic_info
from src.modeling.classifier import train_classifier, optimize_thresholds, HIVDataset
from src.evaluation.evaluate import evaluate_classifier
from src.analysis.joint_analysis import topic_signal_association, signal_cooccurrence, temporal_trends
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(config):
    # 1. Load raw data (JSONL)
    raw_files = config['data']['raw_files']
    records = []
    for f in raw_files:
        records.extend(pd.read_json(f, lines=True).to_dict('records'))

    # 2. Preprocess
    logger.info("Preprocessing texts...")
    processed = []
    for rec in records:
        text = rec.get('selftext') or rec.get('body')
        if not text:
            continue
        cleaned = full_preprocess_pipeline(text)
        if cleaned:
            rec['cleaned_text'] = cleaned
            processed.append(rec)
    df = pd.DataFrame(processed)
    logger.info(f"Processed {len(df)} texts.")

    # 3. Load annotations (labels)
    # Assume we have a separate file with id -> labels
    labels_df = pd.read_csv(config['data']['annotation_file'])
    df = df.merge(labels_df, on='id', how='inner')
    label_cols = ['F1','F2','F3','F4','M1','M2','M3','M4']

    # 4. Split data (stratified by year and subreddit)
    # For simplicity, we'll do random split here (in practice use stratified)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1/0.9, random_state=42)

    # 5. Topic Modeling on full corpus (or subset)
    logger.info("Training BERTopic...")
    corpus = df['cleaned_text'].tolist()
    topic_model = train_bertopic(corpus)
    # Assign topics to each text
    topics, _ = topic_model.transform(corpus)
    df['topic_id'] = topics
    topic_info = get_topic_info(topic_model)

    # 6. Train classifier
    logger.info("Training RoBERTa classifier...")
    model, history = train_classifier(
        train_texts=train_df['cleaned_text'].tolist(),
        train_labels=train_df[label_cols].values,
        val_texts=val_df['cleaned_text'].tolist(),
        val_labels=val_df[label_cols].values,
        output_dir=config['output_dir']
    )

    # 7. Evaluate on test set
    # (using DataLoader with trained model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_dataset = HIVDataset(test_df['cleaned_text'].tolist(), test_df[label_cols].values, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    # Optimize thresholds on validation set
    thresholds = optimize_thresholds(model, val_loader, device)
    # Predict test set
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    # Apply thresholds
    y_pred = (all_probs >= np.array(thresholds)).astype(int)
    metrics = evaluate_classifier(all_labels, y_pred, all_probs)

    # 8. Joint Analysis
    # Ensure df has predictions (use model on full corpus)
    # For demonstration, we'll use test set only
    df_pred = pd.DataFrame({
        'id': test_df['id'],
        'topic_id': test_df['topic_id'],
        **{col: y_pred[:, i] for i, col in enumerate(label_cols)}
    })

    topic_signal = topic_signal_association(df_pred, 'topic_id', label_cols)
    cooccur = signal_cooccurrence(df_pred, label_cols)
    temporal = temporal_trends(df_pred, 'year', label_cols)

    # Save outputs
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    topic_signal.to_csv(f"{output_dir}/topic_signal.csv", index=False)
    cooccur.to_csv(f"{output_dir}/cooccurrence.csv", index=False)
    temporal.to_csv(f"{output_dir}/temporal.csv", index=False)
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f)

    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    import yaml
    with open('config/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    run_pipeline(config)
