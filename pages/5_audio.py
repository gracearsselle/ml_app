import asyncio
import logging
import os
import traceback

import streamlit as st
from dotenv import load_dotenv
from gemini_client import GeminiLiveClient
import stt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
for noisy in ("watchdog", "websockets", "httpcore", "httpx", "google.auth"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def check_api_keys():
    missing = []
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        missing.append("GEMINI_API_KEY")
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        missing.append("GROQ_API_KEY")
    if missing:
        st.error(
            f"Missing API keys: {', '.join(missing)}. "
            "Please set them in the `.env` file."
        )
        st.info(
            "- **GEMINI_API_KEY**: Get one at https://aistudio.google.com/app/apikey\n"
            "- **GROQ_API_KEY**: Get one at https://console.groq.com/keys"
        )
        st.stop()


check_api_keys()

st.set_page_config(page_title="Grok Audio Chatbot", page_icon="🎙️", layout="wide")

st.title("🎙️ Grok Audio Chatbot")
st.caption("Gemini 3.1 Flash Live · Groq Whisper STT")

with st.sidebar:
    st.header("Configuration")
    st.markdown(
        "- **Gemini API**: "
        f"{'✅ Configured' if GEMINI_API_KEY and GEMINI_API_KEY != 'your_gemini_api_key_here' else '❌ Missing'}\n"
        "- **Groq API**: "
        f"{'✅ Configured' if GROQ_API_KEY and GROQ_API_KEY != 'your_groq_api_key_here' else '❌ Missing'}"
    )
    st.divider()
    st.markdown("### How to use")
    st.markdown("1. **Text mode**: Type a message and press Enter")
    st.markdown("2. **Microphone**: Click the mic button to record")
    st.markdown("3. **Upload**: Drop an audio file in the uploader")
    st.markdown("4. The bot responds with audio + transcription")
    st.divider()
    st.markdown("### Limits (free tier)")
    st.markdown("- Gemini Live session: **15 min** max")
    st.markdown("- Groq Whisper: file ≤ **25 MB**")
    st.divider()
    with st.expander("Clear conversation"):
        if st.button("Clear all messages"):
            st.session_state.messages = []
            st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mic_key" not in st.session_state:
    st.session_state.mic_key = 0

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("transcript"):
            st.markdown(f"*Transcription:* {msg['transcript']}")
        st.markdown(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/wav")


def run_async(coro):
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=120)


def handle_text_input(prompt: str):
    logger.info("handle_text_input called with prompt: %r", prompt[:80])
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🎙️ Thinking..."):
            try:
                client = GeminiLiveClient(api_key=GEMINI_API_KEY)
                audio_bytes, transcript = run_async(client.chat_text(prompt))
                logger.info(
                    "Got response: audio=%d bytes, transcript=%r",
                    len(audio_bytes) if audio_bytes else 0,
                    transcript[:80] if transcript else None,
                )
            except Exception as e:
                logger.error("Gemini error: %s", traceback.format_exc())
                st.error(f"Gemini API error: {e}")
                return

        if transcript:
            st.markdown(f"*Transcription:* {transcript}")

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": transcript or "(audio response)",
                    "transcript": transcript,
                    "audio": audio_bytes,
                }
            )
        elif transcript:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": transcript,
                    "transcript": transcript,
                    "audio": None,
                }
            )
        else:
            st.warning("No response received from the model.")


def handle_audio_input(audio_bytes: bytes, filename: str):
    logger.info(
        "handle_audio_input called with file: %s (%d bytes)", filename, len(audio_bytes)
    )
    max_size_mb = 25
    if len(audio_bytes) > max_size_mb * 1024 * 1024:
        st.error(
            f"Audio file too large ({len(audio_bytes) / 1024 / 1024:.1f} MB). Max is {max_size_mb} MB."
        )
        return

    with st.spinner("📝 Transcribing audio..."):
        try:
            text = stt.transcribe(audio_bytes, filename)
            logger.info("Transcription result: %r", text[:100] if text else None)
        except Exception as e:
            logger.error("STT error: %s", traceback.format_exc())
            st.error(f"Transcription error: {e}")
            return

    if not text or not text.strip():
        st.warning("Could not transcribe audio. Please try again.")
        return

    st.session_state.messages.append(
        {"role": "user", "content": f"🎤 *You said:* {text}"}
    )

    with st.chat_message("user"):
        st.markdown(f"🎤 *You said:* {text}")

    handle_text_input(text)


prompt = st.chat_input("Type your message...")

mic_audio = st.audio_input("Record from microphone")

if mic_audio is not None:
    mic_key = f"mic_{mic_audio.name}_{mic_audio.size}"
    if mic_key not in st.session_state:
        st.session_state[mic_key] = True
        handle_audio_input(mic_audio.getvalue(), mic_audio.name)

uploaded_file = st.file_uploader(
    "Or upload an audio file",
    type=["webm", "mp3", "wav", "m4a", "ogg", "flac", "opus"],
    help="Upload an audio file for transcription (max 25 MB)",
)

if uploaded_file is not None:
    key = f"processed_{uploaded_file.name}_{uploaded_file.size}"
    if key not in st.session_state:
        st.session_state[key] = True
        handle_audio_input(uploaded_file.getvalue(), uploaded_file.name)

if prompt:
    handle_text_input(prompt)
