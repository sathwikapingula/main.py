import streamlit as st
import requests
import tempfile
import os
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="AI Interview Practice System",
    page_icon="🎤"
)

st.title("🎤 AI Interview Practice System")
st.write("Practice interview questions and get AI-based feedback.")

# -----------------------------
# Load Whisper Model
# -----------------------------

@st.cache_resource
def load_whisper():
    return WhisperModel("base", device="cpu", compute_type="int8")

# -----------------------------
# Interview Type
# -----------------------------

interview_type = st.selectbox(
    "Choose Interview Type",
    [
        "Technical Interview",
        "HR Interview",
        "Python Interview",
        "ECE Interview",
        "CSE Interview"
    ]
)

# -----------------------------
# Question
# -----------------------------

questions = {
    "Technical Interview":
        "Explain the difference between AI and Machine Learning.",

    "HR Interview":
        "Tell me about yourself.",

    "Python Interview":
        "What is the difference between a list and a tuple in Python?",

    "ECE Interview":
        "What is a transistor and where is it used?",

    "CSE Interview":
        "What is the difference between a process and a thread?"
}

question = questions[interview_type]

st.subheader("Interview Question")
st.info(question)

# -----------------------------
# Audio Upload
# -----------------------------

audio_file = st.file_uploader(
    "Upload your interview answer",
    type=["wav", "mp3", "m4a"]
)

# -----------------------------
# Process Audio
# -----------------------------

if audio_file is not None:

    if st.button("Analyze My Answer"):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(audio_file.name)[1]
        ) as temp_audio:

            temp_audio.write(audio_file.read())
            audio_path = temp_audio.name

        st.write("🎧 Converting your answer to text...")

        try:
            model = load_whisper()

            segments, info = model.transcribe(audio_path)

            text = ""

            for segment in segments:
                text += segment.text + " "

            text = text.strip()

            st.subheader("📝 Your Answer")
            st.write(text)

            # -----------------------------
            # AI Feedback using Ollama
            # -----------------------------

            prompt = f"""
You are an AI interview evaluator.

Interview question:
{question}

Candidate answer:
{text}

Give feedback in this format:

Score: /10

Strengths:
- 

Weaknesses:
- 

Suggestions:
- 

Improved Answer:
"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False
                }
            )

            if response.status_code == 200:

                result = response.json()

                st.subheader("🤖 AI Feedback")
                st.write(result["response"])

            else:
                st.error(
                    "Ollama is not running. Start Ollama and try again."
                )

        except Exception as e:

            st.error(f"Error: {e}")

        finally:

            if os.path.exists(audio_path):
                os.remove(audio_path)