---
title: SEMA
emoji: 📈
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
license: mit
short_description: 'SEMA: Analyze sentence similarity, sentiment, and more'
---
# 🧪 SEMA: Semantic Evaluation & Matching Analyzer

SEMA is an interactive Gradio-powered NLP tool for analyzing semantic similarity, paraphrasing, sentiment, and linguistic features between two sentences.

## 🚀 Features

- ✅ Sentence similarity scoring (BERT/SBERT models)
- 📊 Visual similarity charts (Bar, Gauge, Heatmap)
- 🔄 Paraphrase detection and generation
- 💬 Sentiment analysis (transformers-based)
- 🧠 Named Entity Recognition (NER)
- 🧾 POS tagging with visual distribution (Pie chart)
- 📚 Text statistics: word counts, overlap, character counts

## 🔧 Models Used

- Sentence Transformers:
  - `all-MiniLM-L6-v2`
  - `paraphrase-MiniLM-L3-v2`
  - `paraphrase-multilingual-MiniLM-L12-v2`
  - `distilbert-base-nli-mean-tokens`
- Paraphraser: `ramsrigouthamg/t5_paraphraser`
- Sentiment: Hugging Face sentiment-analysis pipeline
- NER/POS: spaCy (`en_core_web_trf`)

## 🛠️ Setup

```bash
git clone https://github.com/adityasync/SEMA.git
cd SEMA
pip install -r requirements.txt
python app.py

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference