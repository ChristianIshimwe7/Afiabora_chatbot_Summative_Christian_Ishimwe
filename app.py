import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import traceback

st.set_page_config(page_title="Afiabora-Med", page_icon="🤰")
st.title("🤰 Afiabora-Med: Maternal & Newborn Health Assistant")
st.write("Ask questions about pregnancy, newborn care, and congenital anomaly prevention.")

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Use appropriate dtype
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        
        # Load LoRA adapters from local folder
        lora_path = os.path.join(os.path.dirname(__file__), "afiabora-lora")
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA folder not found at {lora_path}")
        
        model = PeftModel.from_pretrained(base_model, lora_path)
        model.eval()
        
        st.success("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.code(traceback.format_exc())
        st.stop()

# Load model
tokenizer, model = load_model()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    prompt_text = f"""### Instruction:
You are Afiabora-Med, a maternal and child health assistant. Provide accurate, helpful information based on WHO and Rwanda MOH guidelines.

### Input:
{prompt}

### Response:
"""
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
