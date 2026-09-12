# Fine-Tuned DistilBERT IMDB Sentiment Classifier

A fine-tuned DistilBERT transformer model for binary sentiment classification of English movie reviews from the IMDB dataset.

## Project Overview

This project fine-tunes the pretrained `distilbert-base-uncased` model for binary sentiment classification.

The goal is to classify a movie review as:

- **0 → Negative**
- **1 → Positive**

The project also compares the Transformer model against a traditional NLP baseline using TF-IDF + Logistic Regression.

## Project Pipeline

```text
IMDB Dataset
     ↓
Exploratory Data Analysis
     ↓
Train / Validation / Test Split
     ↓
DistilBERT Tokenization
     ↓
Pretrained DistilBERT
     ↓
Fine-Tuning
     ↓
Evaluation
     ↓
Confusion Matrix
     ↓
Error Analysis
     ↓
TF-IDF + Logistic Regression Baseline
     ↓
Model Saving
     ↓
Hugging Face Model Hub
     ↓
Streamlit Deployment
```

## Dataset

The project uses the **IMDB movie review dataset** from Hugging Face.

### Dataset Splits

- **Training set:** 25,000 reviews
- **Test set:** 25,000 reviews
- **Unsupervised set:** 50,000 reviews

The original training set was further split into:

- **Training:** 22,500 reviews
- **Validation:** 2,500 reviews

The final 25,000-review test set was kept separate for final evaluation.

### Features

- `text` → Movie review
- `label` → Sentiment class

### Label Mapping

- `0` → Negative
- `1` → Positive

## Exploratory Data Analysis

The dataset was checked for class distribution, sample reviews, dataset features, and text length.

### Training Review Statistics

- **Average word count:** ~233.8
- **Minimum word count:** 10
- **Maximum word count:** 2,470

### Tokenized Review Statistics

- **Average token count:** ~313.9
- **Minimum token count:** 13
- **Maximum token count:** 3,127

A maximum sequence length of **256 tokens** was used during model training, with truncation applied to longer reviews and padding used for shorter sequences.

## Model

### Pretrained Model

`distilbert-base-uncased`

DistilBERT provides pretrained language representations that were adapted to the sentiment classification task through fine-tuning.

### Classification Architecture

```text
Review
  ↓
Tokenizer
  ↓
Input IDs + Attention Mask
  ↓
DistilBERT
  ↓
Classification Head
  ↓
Logits
  ↓
Softmax
  ↓
Class Probabilities
  ↓
Negative / Positive

The model was configured for two labels.
```
## Training

The Hugging Face `Trainer` API was used to fine-tune the model.

### Training Configuration

- **Epochs:** 2
- **Learning rate:** `2e-5`
- **Training batch size:** 16
- **Evaluation:** After each epoch
- **Best model selection metric:** F1-score

The fine-tuning process was performed on a **Tesla T4 GPU** using Google Colab.

## Final Test Results

The fine-tuned model was evaluated on the held-out 25,000-review test set.

| Metric | Result |
|---|---:|
| Accuracy | **91.36%** |
| Precision | **91.04%** |
| Recall | **91.74%** |
| F1-score | **91.39%** |

## Baseline Comparison

A traditional NLP baseline was implemented using **TF-IDF features with Logistic Regression**.

Both models were evaluated on the same held-out test set.

| Model | Accuracy | F1-score |
|---|---:|---:|
| TF-IDF + Logistic Regression | 89.24% | 89.28% |
| Fine-Tuned DistilBERT | **91.36%** | **91.39%** |

### Improvement

Compared with the TF-IDF baseline:

- **Accuracy improved by 2.12 percentage points**
- **F1-score improved by 2.11 percentage points**

The fine-tuned DistilBERT model performed better than the TF-IDF baseline on the held-out test set.

## Confusion Matrix

The final test-set confusion matrix was:

```text
[[11397, 1103],
 [ 1058, 11442]]
```
#### Interpretation:

- **True Negative: 11,397**
- **False Positive: 1,103**
- **False Negative: 1,058**
- **True Positive: 11,442**

The two major error types were relatively balanced.

## Error Analysis

A total of **2,161** test reviews were misclassified.

Several misclassified examples were inspected manually to understand the common sources of prediction errors.

### Observed Error Patterns

- Mixed sentiment within the same review
- Strong positive phrases inside an overall negative review
- Ambiguous wording
- Context-dependent expressions
- Reviews containing both praise and criticism

For example, some negative reviews contained phrases such as `"superb"`, `"fell in love with"`, or `"favorite"`. These phrases can provide strong positive sentiment cues even when the overall review is negative.

## Inference

The trained model was loaded from the saved checkpoint and used to classify new movie reviews.

### Example

```text
Input:
This movie was absolutely amazing. I loved every minute of it.

Prediction:
Positive

Confidence:
99.59%

The model was also tested with a clearly negative review and correctly predicted the negative sentiment.
```
## Hugging Face Model

The fine-tuned DistilBERT model and tokenizer are hosted on the Hugging Face Model Hub.

**[View Model on Hugging Face](https://huggingface.co/Oblivion22/distilbert-imdb-sentiment)**

## Live Demo

The model is deployed as an interactive Streamlit application.

**[Try the Live Demo](https://distilbert-imdb-sentiment.streamlit.app/)**

The application allows users to enter a movie review and receive:

- Predicted sentiment
- Confidence score
## Repository Structure

```text
distilbert-imdb-sentiment/
│
├── app.py
├── requirements.txt
├── README.md
└── DistilBERT_Sentiment_Classifier.ipynb
```
## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Google Colab
- GitHub
- Hugging Face Hub

## Limitations

- The model was fine-tuned specifically on IMDB movie reviews and may perform differently on text from other domains.
- Mixed sentiment, sarcasm, ambiguity, and context-heavy reviews can still lead to incorrect predictions.
- Reviews longer than 256 tokens are truncated during inference.
- Confidence values are model-generated softmax probabilities and should not be interpreted as guaranteed correctness.

## Future Improvements

- Experiment with dynamic padding to improve training efficiency.
- Compare different maximum sequence lengths such as 128 and 256.
- Perform more systematic error analysis.
- Evaluate the model on sentiment data from domains other than movie reviews.

## Author

**Kartik Kumar**

GitHub: [kartik2006-del](https://github.com/kartik2006-del)
