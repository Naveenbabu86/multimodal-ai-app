import tempfile
import io
import re
import streamlit as st
from PIL import Image
from gtts import gTTS
import pytesseract
import torch

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)

# =====================================================
# TESSERACT PATH
# =====================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =====================================================
# LOAD BLIP MODEL (IMAGE CAPTIONING)
# =====================================================

@st.cache_resource
def load_caption_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    return processor, model

processor, model = load_caption_model()

# =====================================================
# LOAD SPEECH MODEL (LIGHTWEIGHT)
# =====================================================

@st.cache_resource
def load_speech_model():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny"
    )

speech_pipeline = load_speech_model()

# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(page_title="AI Multimedia Generator", layout="centered")
st.title("AI Multimedia Generator")

# =====================================================
# SIDEBAR
# =====================================================

INPUT_TYPES = ["Text", "Image", "Audio"]
OUTPUT_TYPES = ["Text", "Audio", "Image"]

input_type = st.sidebar.selectbox("Choose Input Type", INPUT_TYPES)
selected_outputs = st.sidebar.multiselect("Choose Output Types", OUTPUT_TYPES, default=["Text"])

# =====================================================
# CLEAN TEXT (OCR)
# =====================================================

def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9:/\-\n ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================================================
# TEXT TO AUDIO
# =====================================================

def generate_audio(text):
    tts = gTTS(text=text, lang="en")
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

# =====================================================
# INPUT VARIABLES
# =====================================================

input_text = ""
uploaded_image = None
uploaded_audio = None

# =====================================================
# TEXT INPUT
# =====================================================

if input_type == "Text":
    input_text = st.text_area("Enter Text")

# =====================================================
# IMAGE INPUT
# =====================================================

elif input_type == "Image":
    uploaded_image = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# =====================================================
# AUDIO INPUT
# =====================================================

elif input_type == "Audio":
    uploaded_audio = st.file_uploader("Upload Audio", type=["mp3", "wav"])

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

# =====================================================
# GENERATE OUTPUTS
# =====================================================

if st.button("Generate Outputs"):

    final_text = ""

    # =================================================
    # TEXT INPUT
    # =================================================

    if input_type == "Text":
        final_text = input_text

    # =================================================
    # IMAGE INPUT (OCR + CAPTION)
    # =================================================

    elif input_type == "Image" and uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        gray = image.convert("L")

        raw_text = pytesseract.image_to_string(
            gray,
            config="--oem 3 --psm 4"
        )

        cleaned_text = clean_text(raw_text)

        if len(cleaned_text) > 15:
            final_text = cleaned_text
        else:
            with st.spinner("Analyzing image..."):
                inputs = processor(image, return_tensors="pt")
                output = model.generate(**inputs, max_new_tokens=30)

                caption = processor.decode(
                    output[0],
                    skip_special_tokens=True
                )

                final_text = caption

    # =================================================
    # AUDIO INPUT (LIGHTWEIGHT SPEECH TO TEXT)
    # =================================================

    elif input_type == "Audio" and uploaded_audio is not None:

        import tempfile

        with st.spinner("Converting audio to text..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(uploaded_audio.read())
                temp_audio_path = temp_audio.name

            # TRY FAST MODEL
            try:
                result = speech_pipeline(temp_audio_path)
                final_text = result["text"]

            # FALLBACK MODEL
            except Exception:
                st.warning("Fast model failed, using fallback model...")

                fallback = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-base"
                )

                result = fallback(temp_audio_path)
                final_text = result["text"]

    # =================================================
    # TEXT OUTPUT
    # =================================================

    if "Text" in selected_outputs:
        st.subheader("Text Output")
        st.text_area("Result", final_text, height=250)

    # =================================================
    # IMAGE OUTPUT
    # =================================================

    if "Image" in selected_outputs:
        st.subheader("Image Output")

        if input_type == "Image" and uploaded_image is not None:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            st.info("Image output available only for image input.")

    # =================================================
    # AUDIO OUTPUT
    # =================================================

    if "Audio" in selected_outputs:
        st.subheader("Audio Output")

        audio_buffer = generate_audio(final_text)

        st.audio(audio_buffer, format="audio/mp3")

        st.download_button(
            "Download Audio",
            audio_buffer.getvalue(),
            "output_audio.mp3",
            "audio/mpeg"
        )