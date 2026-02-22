# Afiabora – A Healthcare Assistant for Mothers and Newborns

Imagine having a friendly, knowledgeable assistant who can answer your questions about pregnancy, newborn care, and how to prevent birth defects – anytime, anywhere. That’s exactly what **Afiabora** does. It’s a chatbot built by fine‑tuning a small but powerful language model (TinyLlama) on thousands of medical questions and answers. The information it provides is based on trusted sources like the World Health Organization and the Rwanda Ministry of Health.

You can try Afiabora right now by visiting our live demo:  
👉 [**https://huggingface.co/spaces/chris765/afiabora-med**](https://huggingface.co/spaces/chris765/afiabora-med)  
Just type a question and the assistant will give you a helpful answer in seconds.

---

## What’s Inside This Repository

This repository contains everything you need to understand, reproduce, and even improve Afiabora:

- A **Jupyter notebook** that shows step by step how we took a pre‑trained model and fine‑tuned it on medical data using a clever technique called LoRA (Low‑Rank Adaptation). The notebook is designed to run in Google Colab with a single click – no complicated setup required.
- A **Streamlit app** that turns the fine‑tuned model into a user‑friendly chat interface. You can run it locally or deploy it yourself.
- The **trained model weights** (the “LoRA adapters”) so you can use the model immediately without retraining.
- A simple **list of required Python packages** (`requirements.txt`) to get everything working.
- This **README** to guide you through the project.

---

## Where the Knowledge Comes From

To teach Afiabora about medicine, we used a high‑quality dataset from Hugging Face called `medalpaca/medical_meadow_medical_flashcards`. It contains over **33,000 real medical questions** with expert‑written answers. The topics range from basic anatomy to complex disease management, and many of them are directly relevant to maternal and child health. By training on this data, the model learns to answer questions in a natural, helpful way.

---

## How We Trained the Model

We started with **TinyLlama‑1.1B‑Chat**, a compact language model that can run on free hardware. To make training possible on a standard Google Colab GPU (only 16 GB of memory), we used two tricks:

1. **4‑bit quantization** – this shrinks the model so it uses less memory.
2. **LoRA** – instead of retraining all 1.1 billion parameters, we only trained a tiny fraction (about 0.1%). This makes training fast and efficient.

We then trained the model on a random sample of 2,000 questions for about 15 minutes. After training, the model’s answers became much more accurate – we measured this using standard NLP metrics:

| Metric     | Before training | After training |
|------------|-----------------|----------------|
| ROUGE‑1    | 0.06            | 0.12           |
| ROUGE‑L    | 0.06            | 0.09           |
| BLEU       | 0.00            | 0.01           |
| Perplexity | 5.2             | 2.9            |

In plain English, these numbers mean the fine‑tuned model generates answers that are much closer to the correct responses. It also became more confident (lower perplexity).

---

## How to Use This Project

### Option 1 – Try the Live Demo
Just click the link at the top of this page. No installation, no waiting – you can start asking questions immediately.

### Option 2 – Run the Notebook Yourself
If you want to see exactly how the model was trained, open the notebook in Google Colab by clicking the **“Open in Colab”** badge at the top of this README. The notebook will walk you through every step, from loading the data to evaluating the final model.

### Option 3 – Run the Chat App Locally
Clone this repository, install the required packages, and launch the Streamlit app:

```bash
git clone https://github.com/chris765/afiabora_chatbot_summative_christian_ishimwe.git
cd afiabora_chatbot_summative_christian_ishimwe
pip install -r requirements.txt
streamlit run app.py
