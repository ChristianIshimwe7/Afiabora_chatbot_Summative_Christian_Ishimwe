import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import traceback

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="Afiabora-Med",
    page_icon="🤰",
    layout="centered"
)

st.title("🤰 Afiabora-Med: Maternal & Newborn Health Assistant")
st.markdown("""
Ask questions about pregnancy, newborn care, and congenital anomaly prevention.  
Based on WHO and Rwanda Ministry of Health guidelines.
""")

# -------------------- Model Loading (cached) --------------------
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    """Load the tokenizer and fine‑tuned model (CPU only)."""
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

        # ---- Tokenizer ----
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # ---- Base model on CPU with float32 ----
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        # ---- Load LoRA adapters from local folder ----
        lora_path = os.path.join(os.path.dirname(__file__), "afiabora-lora")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA folder not found at {lora_path}. "
                                    "Please ensure 'afiabora-lora' is in the same directory as app.py.")

        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()

        st.success("✅ Model loaded successfully!")
        return tokenizer, model

    except Exception as e:
        st.error(f"❌ Failed to load model:\n{str(e)}")
        st.code(traceback.format_exc())
        st.stop()

# Load the model (runs once and caches)
tokenizer, model = load_model()

# -------------------- Chat Interface --------------------
# Initialise session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask your question here..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Construct the prompt exactly as during training
    prompt_text = f"""### Instruction:
You are Afiabora-Med, a maternal and child health assistant. Provide accurate, helpful information based on WHO and Rwanda MOH guidelines.

### Input:
{prompt}

### Response:
"""

    # Tokenize and generate
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode and clean the answer
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in full_response:
        answer = full_response.split("### Response:")[-1].strip()
    else:
        answer = full_response.strip()

    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
