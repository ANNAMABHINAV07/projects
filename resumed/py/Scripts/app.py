import os
import warnings
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------- BASIC CONFIG --------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="JARVIS • Personal AI",
    page_icon="🦾",
    layout="centered"
)

# -------------------- IRON MAN / JARVIS THEME --------------------
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #1a0000 0%, #000000 65%);
        color: #f5c77a;
        font-family: 'Segoe UI', sans-serif;
    }

    .jarvis-title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        color: #ff3b3b;
        text-shadow: 0 0 14px rgba(255, 60, 60, 0.9);
        margin-bottom: 0;
    }

    .jarvis-sub {
        text-align: center;
        color: #f5c77a;
        letter-spacing: 3px;
        margin-top: -6px;
        margin-bottom: 25px;
    }

    .stChatMessage[data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, #5c0000, #8b0000);
        border-radius: 14px;
        box-shadow: 0 0 12px rgba(255, 60, 60, 0.45);
    }

    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, #1a1a1a, #2b2b2b);
        border-left: 4px solid #f5c77a;
        border-radius: 14px;
        box-shadow: 0 0 12px rgba(245, 199, 122, 0.3);
    }

    textarea {
        background-color: #0f0f0f !important;
        color: #f5c77a !important;
        border: 1px solid #ff3b3b !important;
        box-shadow: 0 0 10px rgba(255, 60, 60, 0.7);
    }

    button {
        background: linear-gradient(135deg, #b30000, #ff3b3b) !important;
        color: white !important;
        border-radius: 10px !important;
        box-shadow: 0 0 12px rgba(255, 60, 60, 0.9);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
st.markdown("<div class='jarvis-title'>J.A.R.V.I.S</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='jarvis-sub'>JUST A RATHER VERY INTELLIGENT SYSTEM</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border:1px solid #ff3b3b;'>", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -------------------- GENERATION CONFIG --------------------
GENERATION_CONFIG = dict(
    max_new_tokens=120,
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- CLEAR BUTTON --------------------
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🧹 CLEAR"):
        st.session_state.messages = []
        st.rerun()

# -------------------- INTELLIGENCE RULES --------------------
def is_greeting(text):
    return text.strip().lower() in [
        "hi", "hii", "hello", "hey", "hai",
        "hi jarvis", "hello jarvis"
    ]

def is_generic_help(text):
    text = text.lower().strip()
    return text in [
        "help",
        "i need help",
        "need help",
        "can you help me",
        "assist me"
    ]

def is_echo(user_text, model_text):
    return user_text.strip().lower() == model_text.strip().lower()

def build_prompt(user_input):
    return f"""
You are JARVIS, an intelligent and polite AI assistant.

Respond helpfully and clearly.
Do NOT repeat the user's input.
If the input is unclear, ask a clarifying question.

User input:
{user_input}

JARVIS response:
"""

# -------------------- DISPLAY CHAT --------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------- INPUT --------------------
user_input = st.chat_input("Speak, Sir…")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # ---- CONTROL LAYER ----
    if is_greeting(user_input):
        response = "Hello Sir. JARVIS online and ready. How may I assist you?"

    elif is_generic_help(user_input):
        response = (
            "Of course, Sir. Please tell me what you need help with — "
            "learning, technical guidance, or anything else."
        )

    else:
        prompt = build_prompt(user_input)
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(**inputs, **GENERATION_CONFIG)

        response = tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()

        if is_echo(user_input, response):
            response = "Understood, Sir. Could you please explain a little more?"

        if not response:
            response = "I’m afraid I didn’t catch that, Sir."

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()
