# HIV‑Reddit‑NLP: Analysis of Fatigue and Medication Discourse

This repository accompanies the paper *"NLP‑Based Analysis of HIV Fatigue and Medication Discourse on Reddit: A Longitudinal Study (2015–2025)"*. It provides a reproducible end‑to‑end pipeline for:

- Collecting Reddit posts and comments from r/hivaids and r/HIV (2015‑2025)
- Preprocessing and cleaning social media text
- Manually annotating a stratified sample with 8 clinical labels (fatigue subtypes and medication adjustment)
- Discovering latent topics with BERTopic
- Fine‑tuning a RoBERTa‑based multi‑label classifier
- Evaluating against baselines and performing ablation studies
- Joint analysis of topic‑signal associations, co‑occurrence, and temporal trends

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hiv-reddit-nlp.git
   cd hiv-reddit-nlp
