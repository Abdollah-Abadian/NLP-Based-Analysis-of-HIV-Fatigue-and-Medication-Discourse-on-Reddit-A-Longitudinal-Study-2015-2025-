import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.metrics import jaccard_score
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def topic_signal_association(
    df: pd.DataFrame,
    topic_col: str = 'topic_id',
    label_cols: List[str] = None
) -> pd.DataFrame:
    """
    For each topic, compute prevalence of each signal and chi‑square test.
    df must contain topic assignments and binary label columns.
    Returns DataFrame with topic, signal, prevalence, chi2, p.
    """
    if label_cols is None:
        label_cols = [f'F{i}' for i in range(1,5)] + [f'M{i}' for i in range(1,5)]
    results = []
    topics = df[topic_col].unique()
    for topic in topics:
        topic_mask = df[topic_col] == topic
        n_topic = topic_mask.sum()
        for lbl in label_cols:
            pos_count = df[lbl][topic_mask].sum()
            preval = pos_count / n_topic if n_topic > 0 else 0.0
            # Chi‑square with rest of corpus
            rest_mask = ~topic_mask
            rest_pos = df[lbl][rest_mask].sum()
            rest_neg = rest_mask.sum() - rest_pos
            topic_pos = pos_count
            topic_neg = n_topic - topic_pos
            contingency = np.array([[topic_pos, topic_neg],
                                    [rest_pos, rest_neg]])
            if contingency.sum() > 0:
                chi2, p, _, _ = chi2_contingency(contingency)
            else:
                chi2, p = np.nan, np.nan
            results.append({
                'topic': topic,
                'signal': lbl,
                'prevalence': preval,
                'chi2': chi2,
                'p_value': p
            })
    return pd.DataFrame(results)


def signal_cooccurrence(
    df: pd.DataFrame,
    label_cols: List[str]
) -> pd.DataFrame:
    """Compute Jaccard similarity for all pairs of signals."""
    results = []
    for i, c1 in enumerate(label_cols):
        for c2 in label_cols[i+1:]:
            y1 = df[c1].values
            y2 = df[c2].values
            # Jaccard for binary vectors
            intersection = np.logical_and(y1, y2).sum()
            union = np.logical_or(y1, y2).sum()
            jaccard = intersection / union if union > 0 else 0.0
            results.append({
                'signal_1': c1,
                'signal_2': c2,
                'jaccard': jaccard
            })
    return pd.DataFrame(results).sort_values('jaccard', ascending=False)


def temporal_trends(
    df: pd.DataFrame,
    date_col: str = 'year',
    label_cols: List[str] = None
) -> pd.DataFrame:
    """Compute annual prevalence of each signal."""
    if label_cols is None:
        label_cols = [f'F{i}' for i in range(1,5)] + [f'M{i}' for i in range(1,5)]
    annual = df.groupby(date_col).agg(
        total_texts=('id', 'count'),
        **{lbl: (lbl, 'sum') for lbl in label_cols}
    ).reset_index()
    for lbl in label_cols:
        annual[f'{lbl}_prev'] = annual[lbl] / annual['total_texts']
    return annual
