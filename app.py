# app.py - Afiabora Healthcare Assistant Flask App

import torch
from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- Configuration ----------
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "./afiabora-lora"          # path to your saved LoRA adapters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load tokenizer and base model ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load the fine‑tuned LoRA adapters
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

# ---------- Chat function (matches your notebook) ----------
def chat(message, history=None):
    prompt = f"""### Instruction:
You are Afiabora-Med, a maternal and child health assistant. Provide accurate, helpful information based on WHO and Rwanda MOH guidelines.

### Input:
{message}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response

# ---------- Flask app ----------
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Afiabora Health Assistant</title>
</head>
<body>
    <h2>Afiabora – Healthcare Assistant</h2>
    <form method="post">
        <label for="question">Your Question:</label><br>
        <input type="text" id="question" name="question" size="50"><br><br>
        <input type="submit" value="Ask">
    </form>
    {% if answer %}
        <h3>Answer:</h3>
        <p>{{ answer }}</p>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    answer = None
    if request.method == 'POST':
        question = request.form['question']
        answer = chat(question)
    return render_template_string(HTML_TEMPLATE, answer=answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
