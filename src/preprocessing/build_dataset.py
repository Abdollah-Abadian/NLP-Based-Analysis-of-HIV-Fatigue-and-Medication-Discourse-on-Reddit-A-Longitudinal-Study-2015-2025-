"""Build the full corpus Parquet dataset from raw JSONL files.

Usage:
    python -m src.data_preparation.build_dataset \
        --raw-dir data/raw/ \
        --output-dir data/processed/
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
import logging

# Import the preprocessing pipeline from the earlier module
from src.preprocessing.preprocess import full_preprocess_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def load_jsonl_files(file_patterns: list) -> pd.DataFrame:
    """Load multiple JSONL files and concatenate into a single DataFrame."""
    all_records = []
    for pattern in file_patterns:
        path = Path(pattern)
        if not path.exists():
            logger.warning(f"File not found: {path}. Skipping.")
            continue
        logger.info(f"Loading {path}...")
        for line in tqdm(path.open(encoding='utf-8'), desc=f"Reading {path.name}"):
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(all_records)


def build_dataset(raw_dir: str, output_dir: str):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load all raw JSONL sources
    file_patterns = [
        raw_path / "arctic_shift_hivaids.jsonl",
        raw_path / "arctic_shift_hiv.jsonl",
        raw_path / "pushshift_patched_hivaids.jsonl",
        raw_path / "pushshift_patched_hiv.jsonl",
        raw_path / "praw_hivaids_2023_2025.jsonl",
        raw_path / "praw_hiv_2023_2025.jsonl",
    ]
    df = load_jsonl_files(file_patterns)
    logger.info(f"Loaded {len(df)} raw records.")

    # 2. Deduplicate by id (keep first occurrence)
    df = df.drop_duplicates(subset='id', keep='first')
    logger.info(f"After deduplication: {len(df)} records.")

    # 3. Extract text field (posts use 'selftext', comments use 'body')
    df['original_text'] = df.apply(
        lambda row: row.get('selftext') if row.get('record_type') == 'post' else row.get('body'),
        axis=1
    )
    # Drop records with no text
    df = df[df['original_text'].notna() & (df['original_text'].str.strip() != '')]
    logger.info(f"After dropping empty texts: {len(df)} records.")

    # 4. Apply preprocessing pipeline
    logger.info("Applying preprocessing pipeline...")
    df['cleaned_text'] = df['original_text'].apply(
        lambda x: full_preprocess_pipeline(x, filter_english_flag=True)
    )
    df = df[df['cleaned_text'].notna()]
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    logger.info(f"After preprocessing: {len(df)} records.")

    # 5. Extract temporal features
    df['year'] = pd.to_datetime(df['created_utc'], unit='s').dt.year
    df['month'] = pd.to_datetime(df['created_utc'], unit='s').dt.month
    df['created_utc'] = df['created_utc'].astype(float)

    # 6. Ensure consistent metadata_source (fill missing)
    if 'metadata_source' not in df.columns:
        df['metadata_source'] = 'unknown'
    else:
        df['metadata_source'] = df['metadata_source'].fillna('unknown')

    # 7. Select final columns for the corpus
    final_columns = [
        'id', 'subreddit', 'record_type', 'year', 'month', 'created_utc',
        'score', 'num_comments', 'author', 'original_text', 'cleaned_text',
        'word_count', 'metadata_source'
    ]
    df = df[final_columns]

    # 8. Write to Parquet
    corpus_path = out_path / 'full_corpus.parquet'
    df.to_parquet(corpus_path, index=False)
    logger.info(f"Saved full corpus ({len(df)} rows) to {corpus_path}")

    # 9. Write lighter version (just id + cleaned_text) for model input
    text_path = out_path / 'preprocessed_texts.parquet'
    df[['id', 'cleaned_text']].to_parquet(text_path, index=False)
    logger.info(f"Saved preprocessed texts to {text_path}")

    # 10. Write manifest
    manifest = {
        'created_at': datetime.now().isoformat(),
        'total_records': len(df),
        'posts': int(df['record_type'].eq('post').sum()),
        'comments': int(df['record_type'].eq('comment').sum()),
        'subreddit_counts': df['subreddit'].value_counts().to_dict(),
        'year_counts': df['year'].value_counts().sort_index().to_dict(),
    }
    with open(out_path / 'dataset_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    logger.info("Dataset manifest saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', required=True, help='Directory containing raw JSONL files')
    parser.add_argument('--output-dir', required=True, help='Directory to save processed Parquet')
    args = parser.parse_args()
    build_dataset(args.raw_dir, args.output_dir)
