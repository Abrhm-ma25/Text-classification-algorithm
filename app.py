from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import gradio as gr

# Path to your exported model
MODEL_PATH = "Model/"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Label mapping (same order you trained)
label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}


# ----------- Single text prediction -----------
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = torch.argmax(logits, dim=1).item()
    return label_map[pred_id]


# ----------- CSV batch prediction -----------
def predict_csv(file):
    df = pd.read_csv(file.name)

    if "text" not in df.columns:
        return pd.DataFrame({"error": ["CSV must contain a 'text' column."]})

    preds = []
    for txt in df["text"]:
        preds.append(predict_sentiment(txt))

    df["predicted_sentiment"] = preds
    return df


# ----------- Gradio UI (Blocks) -----------
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Twitter Entity Sentiment Classifier (RoBERTa-base)")

    with gr.Tab("🔍 Single Text Prediction"):
        text_input = gr.Textbox(lines=3, placeholder="Type a tweet...")
        text_output = gr.Label()
        gr.Button("Predict").click(
            predict_sentiment,
            inputs=text_input,
            outputs=text_output
        )

    with gr.Tab("📁 CSV Batch Prediction"):
        csv_input = gr.File(label="Upload CSV (must contain a 'text' column)")
        csv_output = gr.Dataframe()
        gr.Button("Predict CSV").click(
            predict_csv,
            inputs=csv_input,
            outputs=csv_output
        )

demo.launch(share=True)

