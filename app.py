import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import torch
import spacy
from transformers import pipeline, AutoModelForSeq2SeqLM, T5Tokenizer
import functools

# Model Caching
@functools.lru_cache(maxsize=1)
def load_sentence_model(name):
    return SentenceTransformer(name)

@functools.lru_cache(maxsize=1)
def load_paraphraser():
    tokenizer = T5Tokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
    model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@functools.lru_cache(maxsize=1)
def load_sentiment():
    return pipeline("sentiment-analysis")

# Load static models
model = load_sentence_model('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_trf")
paraphraser = load_paraphraser()
sentiment = load_sentiment()

# Similarity and Visualization
def get_similarity(sentence1, sentence2, model_name, visualization_type):
    model_local = load_sentence_model(model_name)
    emb1 = model_local.encode(sentence1, convert_to_tensor=True)
    emb2 = model_local.encode(sentence2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()

    if visualization_type == "Bar Chart":
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Similarity'], [score], color='#4CAF50', edgecolor='black')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Cosine Similarity')
        ax.text(0, score + 0.03, f'{score:.2f}', ha='center', fontsize=12, fontweight='bold')

    elif visualization_type == "Gauge":
        fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={'projection': 'polar'})
        theta = np.linspace(0, np.pi, 100)
        ax.plot(theta, [1] * 100, color='lightgray', linewidth=20, alpha=0.5)
        ax.plot(theta[:int(score * 100)], [1] * int(score * 100), color='#2196F3', linewidth=20)
        ax.set_ylim(0, 1.2)
        ax.set_axis_off()
        ax.text(0, 0, f'{score:.2f}', ha='center', va='center', fontsize=18, fontweight='bold')

    else:  # Heatmap
        fig, ax = plt.subplots(figsize=(3, 3))
        cax = ax.imshow([[score]], cmap='coolwarm', vmin=0, vmax=1)
        fig.colorbar(cax, orientation='vertical')
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0, 0, f'{score:.2f}', ha='center', va='center', fontsize=18, color='black', fontweight='bold')

    return score, f"Similarity Score: {score:.4f}", fig

# Text Analysis
def analyze_text(sentence1, sentence2):
    s1_words, s2_words = len(sentence1.split()), len(sentence2.split())
    s1_chars, s2_chars = len(sentence1), len(sentence2)
    common = set(sentence1.lower().split()).intersection(set(sentence2.lower().split()))
    overlap = len(common)/max(len(set(sentence1.lower().split())), len(set(sentence2.lower().split())))
    return f"""
## Text Analysis
**Sentence 1:** {s1_words} words, {s1_chars} characters  
**Sentence 2:** {s2_words} words, {s2_chars} characters  
**Common Words:** {', '.join(common) if common else 'None'}  
**Word Overlap Rate:** {overlap:.2f}
"""

# Named Entity Recognition
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# POS Tagging
def get_pos_tags(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

def plot_pos_tags(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    def count_pos(doc):
        counts = {}
        for token in doc:
            counts[token.pos_] = counts.get(token.pos_, 0) + 1
        return counts

    pos_counts1 = count_pos(doc1)
    pos_counts2 = count_pos(doc2)

    # Combine counts for pie chart
    combined_counts = {}
    for tag in set(pos_counts1) | set(pos_counts2):
        combined_counts[tag] = pos_counts1.get(tag, 0) + pos_counts2.get(tag, 0)

    labels = list(combined_counts.keys())
    sizes = list(combined_counts.values())

    # Colors sampled to match your uploaded pie chart visually
    custom_colors = [
        '#000066',  # Deep navy (N_SING)
        '#CCCCFF',  # Light lavender (P)
        '#0066CC',  # Blue (DELM)
        '#FF9999',  # Light red (ADJ_SIM)
        '#660066',  # Deep purple (CON)
        '#CCFFFF',  # Light cyan (N_PL)
        '#FFFFCC',  # Light yellow (V_PA)
        '#990033',  # Deep rose (PRO)
        '#9999FF',  # Light blue/purple (ETC)
        '#9966FF',  # Extra if needed
        '#CC66CC'   # Extra if needed
    ]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=custom_colors[:len(sizes)])
    ax.axis('equal')  # Equal aspect ratio makes the pie circular.
    ax.set_title("Combined POS Tag Distribution")

    return fig



# Paraphrase Detection
def detect_paraphrase(score, threshold=0.8):
    return "✅ Likely Paraphrase" if score >= threshold else "❌ Not a Paraphrase"

# Paraphrase Generator
def generate_paraphrases(text):
    try:
        outputs = paraphraser(text, max_length=60, num_return_sequences=2, do_sample=True)
        return [o['generated_text'] for o in outputs]
    except:
        return ["Paraphrasing failed or model not loaded."]

# Sentiment
def get_sentiment(text):
    try:
        return sentiment(text)[0]
    except:
        return {'label': 'Unknown', 'score': 0.0}

# Main processing
def process_text(sentence1, sentence2, model_name, visualization_type, perform_analysis, compare_dataset):
    outputs = []

    score, score_text, fig = get_similarity(sentence1, sentence2, model_name, visualization_type)
    outputs.extend([score_text, fig])

    analysis = analyze_text(sentence1, sentence2) if perform_analysis else ""
    outputs.append(analysis)

    paraphrase_result = detect_paraphrase(score)
    outputs.append(paraphrase_result)

    ner1 = extract_entities(sentence1)
    ner2 = extract_entities(sentence2)
    ner_display = f"""
## Named Entities

**Sentence 1:** {', '.join([f'{e[0]} ({e[1]})' for e in ner1]) if ner1 else 'None'}  
**Sentence 2:** {', '.join([f'{e[0]} ({e[1]})' for e in ner2]) if ner2 else 'None'}  
"""
    outputs.append(ner_display)

    s1_sentiment = get_sentiment(sentence1)
    s2_sentiment = get_sentiment(sentence2)
    senti_display = f"""
## Sentiment Analysis

**Sentence 1:** {s1_sentiment['label']} (score: {s1_sentiment['score']:.2f})  
**Sentence 2:** {s2_sentiment['label']} (score: {s2_sentiment['score']:.2f})  
"""
    outputs.append(senti_display)

    para1 = generate_paraphrases(sentence1)
    para2 = generate_paraphrases(sentence2)
    para_text = f"""
## Paraphrase Suggestions

**Sentence 1:**  
- {para1[0]}  
- {para1[1]}

**Sentence 2:**  
- {para2[0]}  
- {para2[1]}
"""
    outputs.append(para_text)

    # POS Tagging
    pos1 = get_pos_tags(sentence1)
    pos2 = get_pos_tags(sentence2)
    pos_text = f"""
## Part-of-Speech (POS) Tags

**Sentence 1:**  
{', '.join([f"{word} ({pos})" for word, pos in pos1])}

**Sentence 2:**  
{', '.join([f"{word} ({pos})" for word, pos in pos2])}
"""
    outputs.append(pos_text)
    outputs.append(plot_pos_tags(sentence1, sentence2))

    outputs.append("✅ Your input has been submitted! Please check the 📊 Results tab.")
    return outputs

# Models
models = [
    'all-MiniLM-L6-v2',
    'paraphrase-multilingual-MiniLM-L12-v2',
    'paraphrase-MiniLM-L3-v2',
    'distilbert-base-nli-mean-tokens'
]

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧪 SEMA: Semantic Evaluation & Matching Analyzer")
    gr.Markdown("Explore sentence meaning, similarity, and more.")

    with gr.Tabs():
        with gr.Tab("📝 Input"):
            sentence1 = gr.Textbox(label="Sentence 1", lines=4)
            sentence2 = gr.Textbox(label="Sentence 2", lines=4)
            model_name = gr.Dropdown(choices=models, value=models[0], label="Model")
            visualization_type = gr.Radio(["Bar Chart", "Gauge", "Heatmap"], value="Gauge", label="Visualization")
            perform_analysis = gr.Checkbox(label="Extra Text Analysis", value=True)
            compare_dataset = gr.Checkbox(label="Compare with Dataset", value=False)
            submit_btn = gr.Button("Run Analysis")
            status_msg = gr.Textbox(label="Status", interactive=False)

        with gr.Tab("📊 Results"):
            sim_result = gr.Textbox(label="Similarity Score", interactive=False)
            vis_output = gr.Plot(label="Visualization")
            para_result = gr.Textbox(label="Paraphrase Detection", interactive=False)

        with gr.Tab("🔬 Deep Insights"):
            with gr.Accordion("📚 Text Statistics", open=True):
                stats_output = gr.Markdown()
            with gr.Accordion("🧠 Named Entity Recognition", open=False):
                ner_output = gr.Markdown()
            with gr.Accordion("💬 Sentiment Analysis", open=False):
                sentiment_output = gr.Markdown()
            with gr.Accordion("🌀 Paraphrase Suggestions", open=False):
                para_output = gr.Markdown()
            with gr.Accordion("🧾 POS Tagging", open=False):
                pos_output = gr.Markdown()
                pos_plot_output = gr.Plot()

    gr.Examples([
        ["The sky is blue.", "The sky has a beautiful blue color."],
        ["What is your name?", "Can you tell me your name?"]
    ], inputs=[sentence1, sentence2])

    submit_btn.click(
        fn=process_text,
        inputs=[sentence1, sentence2, model_name, visualization_type, perform_analysis, compare_dataset],
        outputs=[
            sim_result,
            vis_output,
            stats_output,
            para_result,
            ner_output,
            sentiment_output,
            para_output,
            pos_output,
            pos_plot_output,
            status_msg
        ]
    )

demo.launch()
