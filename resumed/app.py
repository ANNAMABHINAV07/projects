import os
import warnings
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -------------------- STREAMLIT + TORCH FIX --------------------
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="JARVIS • Personal AI",
    page_icon="🦾",
    layout="centered"
)

# -------------------- IRON MAN UI --------------------
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
    }
    .jarvis-sub {
        text-align: center;
        color: #f5c77a;
        letter-spacing: 3px;
        margin-bottom: 25px;
    }
    .stChatMessage[data-testid="stChatMessage-user"] {
        background: linear-gradient(135deg, #5c0000, #8b0000);
        border-radius: 14px;
    }
    .stChatMessage[data-testid="stChatMessage-assistant"] {
        background: linear-gradient(135deg, #1a1a1a, #2b2b2b);
        border-left: 4px solid #f5c77a;
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='jarvis-title'>J.A.R.V.I.S</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='jarvis-sub'>JUST A RATHER VERY INTELLIGENT SYSTEM</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border:1px solid #ff3b3b;'>", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

GENERATION_CONFIG = dict(
    max_new_tokens=160,
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
    repetition_penalty=1.2
)

# -------------------- SESSION --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------- HELPERS --------------------
def is_greeting(text):
    return text.lower().strip() in ["hi", "hello", "hey", "hi jarvis", "hello jarvis"]

def is_generic_help(text):
    return text.lower().strip() in [
        "help", "i need help", "need help", "assist me"
    ]

def is_follow_up(text):
    return text.lower().strip() in [
        "i need more", "more", "explain more", "tell me more", "more details"
    ]

def build_prompt(user_input, extra_context=None):
    if extra_context:
        return f"""
You are JARVIS, an intelligent assistant.

Previous answer:
{extra_context}

User wants more explanation.

Provide a clearer and more detailed explanation.
"""
    return f"""
You are JARVIS, an intelligent assistant.

Answer clearly and correctly.

Question:
{user_input}

Answer:
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

    if is_greeting(user_input):
        response = "Hello Sir. JARVIS online. How may I assist you?"

    elif is_generic_help(user_input):
        response = "Of course, Sir. Please tell me what you need help with."

    elif is_follow_up(user_input):
        # get last assistant message
        last_answer = next(
            (m["content"] for m in reversed(st.session_state.messages)
             if m["role"] == "assistant"),
            None
        )
        prompt = build_prompt(user_input, last_answer)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, **GENERATION_CONFIG)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    else:
        prompt = build_prompt(user_input)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, **GENERATION_CONFIG)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.strip()}
    )
    st.rerun()
