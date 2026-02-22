# ======================================================
# FULL SETUP: Install, Write App, Launch, and Expose
# ======================================================

import subprocess
import threading
import time
import requests
import os
import sys

# ---- 1. Install required packages (if not already) ----
!pip install -q streamlit pyngrok requests

# ---- 2. Write the Streamlit app (app.py) ----
app_code = '''
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import traceback

st.set_page_config(page_title="Afiabora-Med", page_icon="🤰")
st.title("🤰 Afiabora-Med: Maternal & Newborn Health Assistant")
st.markdown("Ask questions about pregnancy, newborn care, and congenital anomaly prevention.\\nBased on WHO and Rwanda MOH guidelines.")

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        lora_path = os.path.join(os.path.dirname(__file__), "afiabora-lora")
        if not os.path.exists(lora_path):
            st.error(f"LoRA folder not found at {lora_path}. Make sure 'afiabora-lora' is in the same directory.")
            st.stop()

        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        st.success("✅ Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_text = f"""### Instruction:
You are Afiabora-Med, a maternal and child health assistant. Provide accurate, helpful information based on WHO and Rwanda MOH guidelines.

### Input:
{prompt}

### Response:
"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full_response:
        answer = full_response.split("### Response:")[-1].strip()
    else:
        answer = full_response.strip()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
'''

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py written.")

# ---- 3. Check if LoRA folder exists ----
if not os.path.exists("afiabora-lora"):
    print("⚠️  Warning: 'afiabora-lora' folder not found. The app will fail to load the model.")
    print("    Please upload your LoRA adapter folder (afiabora-lora) to the current directory.")
    # Optionally, you can provide a way to download from Hugging Face if you stored it there.
else:
    print("✅ LoRA folder found.")

# ---- 4. Start Streamlit in background ----
def run_streamlit():
    with open("streamlit.log", "w") as log:
        subprocess.Popen(
            ["streamlit", "run", "app.py", "--server.port", "8501"],
            stdout=log,
            stderr=log
        )

thread = threading.Thread(target=run_streamlit)
thread.daemon = True
thread.start()

print("⏳ Waiting for Streamlit to start...")
for i in range(30):   # wait up to 30 seconds
    time.sleep(1)
    try:
        r = requests.get("http://localhost:8501", timeout=1)
        if r.status_code == 200:
            print("✅ Streamlit is up!")
            break
    except:
        pass
else:
    print("❌ Streamlit failed to start. Check streamlit.log for errors.")
    with open("streamlit.log", "r") as log:
        print(log.read())
    # Fallback: offer console chat
    print("\n--- FALLBACK: Console chat ---")
    print("You can still test the model using the console loop below.")
    print("Run the following cell to start a text‑based chat.\n")
    sys.exit()

# ---- 5. Start ngrok tunnel ----
from pyngrok import ngrok

# 🔑 Replace with your own ngrok authtoken (get one free at https://dashboard.ngrok.com)
NGROK_AUTH_TOKEN = "3A23VPnckroypvpv26ukMRPexTq_bK7kyJhPWfRsfityNkxb"   # <-- YOUR TOKEN HERE
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

public_url = ngrok.connect(8501)
print(f"\n🌍 Public URL: {public_url}\n")
print("Share this link – your Streamlit app is live!")
