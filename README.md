# Twitter Sentiment Analysis

This repository contains a complete pipeline for fine-tuning transformer models on a sentiment analysis task using tweets.  
Two architectures are trained and compared:

- RoBERTa-base (encoder-only classifier)
- T5-small (encoder-decoder text-to-text model)

The goal is to identify the best model for identifying the sentiment expressed toward entities (such as people, organizations, or products) in Twitter posts, based on F1-score.

---

## 1. Run on Google Colab

You can execute the full training notebook using the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Abrhm-ma25/Text-classification-algorithm/blob/main/FT_Transformers_.ipynb
)

---

## 2. Project Overview

This project trains machine-learning models to classify tweets into:

- Positive  
- Negative  
- Neutral  

The pipeline includes:

- Data loading  
- Preprocessing  
- Visualization of sentiment distribution  
- Fine-tuning RoBERTa  
- Fine-tuning T5  
- Exact evaluation for both models  
- Refined keyword-based evaluation for T5  
- Model comparison  
- Automatic export of the best model  
- Automatic generation of requirements.txt  

---

## 3. How to Reproduce the Experiments

You can run the notebook either in Colab or locally.

### Option 1 – Run in Google Colab (recommended)

Simply open the notebook using the link above and run all cells.

### Option 2 – Run locally

#### Clone the repository

```bash
git clone https://github.com/Abrhm-ma25/Text-classification-algorithm.git
cd Text-classification-algorithm
```
Create a virtual environment

Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies

Install exact versions used during training:
```bash
pip install -r requirements.txt
```
Run the notebook :
```bash
jupyter notebook
```
Then open:
```bash
FT_Transformers_.ipynb
```

## 4. Contributors to the repository

kevin.dallaporta@edu.dsti.institute

thomas.barzellino@edu.dsti.institute

abraham.ibitowa@edu.dsti.institute


