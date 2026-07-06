import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Download NLTK resources once
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

STOPWORDS = set(stopwords.words('english'))
STOPWORDS.discard('hiv')   # keep domain‑specific term
LEMMATIZER = WordNetLemmatizer()

# Common social‑media abbreviations (expanded)
ABBREV_MAP = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "shouldn't": "should not",
    "couldn't": "could not", "wouldn't": "would not",
    "i'm": "i am", "you're": "you are", "he's": "he is", "she's": "she is",
    "it's": "it is", "we're": "we are", "they're": "they are",
    "i've": "i have", "you've": "you have", "we've": "we have",
    "they've": "they have", "i'll": "i will", "you'll": "you will",
    "he'll": "he will", "she'll": "she will", "we'll": "we will",
    "they'll": "they will", "i'd": "i would", "you'd": "you would",
    "he'd": "he would", "she'd": "she would", "we'd": "we would",
    "they'd": "they would",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "outta": "out of", "kinda": "kind of", "sorta": "sort of",
}


def filter_english(text: str) -> bool:
    """Return True if text is English with high confidence."""
    if not text or len(text) < 10:
        return False
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return False


def clean_text(text: str) -> str:
    """Remove markup, URLs, hashtags, and normalise HTML entities."""
    # Remove Markdown / formatting
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)      # [text](url)
    text = re.sub(r'\*\*?|__|~~', '', text)         # bold/italic/strike
    # Remove URLs and emails
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    # Remove hashtags (keep text)
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_text(text: str) -> str:
    """Lowercase, expand contractions, reduce elongated words."""
    text = text.lower()
    # Expand abbreviations
    for abbr, expansion in ABBREV_MAP.items():
        text = text.replace(abbr, expansion)
    # Reduce elongated words (e.g., "soooo" -> "soo")
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    return text


def remove_stopwords(text: str) -> str:
    """Remove stopwords, but keep 'hiv'."""
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS or t == 'hiv']
    return ' '.join(tokens)


def lemmatize_text(text: str) -> str:
    """Lemmatise tokens with POS tagging."""
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmas = []
    for word, pos in pos_tags:
        if pos.startswith('V'):
            lemma = LEMMATIZER.lemmatize(word, pos='v')
        elif pos.startswith('N'):
            lemma = LEMMATIZER.lemmatize(word, pos='n')
        elif pos.startswith('J'):
            lemma = LEMMATIZER.lemmatize(word, pos='a')
        else:
            lemma = LEMMATIZER.lemmatize(word)
        lemmas.append(lemma)
    return ' '.join(lemmas)


def full_preprocess_pipeline(
    text: str,
    filter_english_flag: bool = True,
    min_length: int = 10
) -> Optional[str]:
    """
    Apply the entire preprocessing chain.
    Returns cleaned text or None if filtering fails.
    """
    if not text or len(text) < min_length:
        return None
    if filter_english_flag and not filter_english(text):
        return None
    text = clean_text(text)
    text = normalize_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    # Remove consecutive spaces again
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) >= 3 else None
