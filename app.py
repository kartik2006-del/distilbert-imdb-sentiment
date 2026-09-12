
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Oblivion22/distilbert-imdb-sentiment"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("IMDB Sentiment Classifier")
st.write("Fine-tuned DistilBERT for movie review sentiment classification.")

text = st.text_area(
    "Enter a movie review:",
    placeholder="Example: This movie was absolutely amazing!"
)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a review.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

        label = "Positive" if predicted_class == 1 else "Negative"

        st.subheader(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2%}")
