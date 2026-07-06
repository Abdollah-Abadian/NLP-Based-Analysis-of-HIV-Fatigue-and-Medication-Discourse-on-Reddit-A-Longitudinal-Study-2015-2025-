from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
import pickle
import logging
from typing import List

logger = logging.getLogger(__name__)


def train_bertopic(
    corpus: List[str],
    min_topic_size: int = 50,
    n_neighbors: int = 15,
    n_components: int = 5,
    cluster_selection_epsilon: float = 0.1
) -> BERTopic:
    """
    Train BERTopic model on the given corpus.
    Returns fitted BERTopic object.
    """
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric='cosine',
        random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric='euclidean',
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True
    )
    vectorizer_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language='english',
        calculate_probabilities=True,
        verbose=True
    )

    topics, probs = topic_model.fit_transform(corpus)
    logger.info(f"BERTopic found {len(topic_model.get_topic_info()) - 1} topics.")
    return topic_model


def get_topic_info(topic_model: BERTopic) -> dict:
    """Return a dict mapping topic_id -> (label, keywords)."""
    topic_df = topic_model.get_topic_info()
    topic_dict = {}
    for _, row in topic_df.iterrows():
        tid = row['Topic']
        if tid != -1:
            keywords = ', '.join([kw for kw, _ in topic_model.get_topic(tid)[:10]])
            topic_dict[tid] = {
                'name': row['Name'],
                'keywords': keywords,
                'size': row['Count']
            }
    return topic_dict
