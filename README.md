 # 📌 AI Multimedia Generator

An intelligent Multimodal AI application built using Streamlit that processes **Text, Image, and Audio inputs** and generates meaningful outputs using AI models.

---

# 🚀 Features

## 📝 Text Input
- Enter text
- Convert to:
  - Speech (Text-to-Speech)
  - Image representation

---

## 🖼️ Image Input
- Extract text using OCR (Tesseract)
- If text is unclear:
  - Generates image caption using BLIP model
- Outputs:
  - Clean text description
  - Audio narration

---

## 🎤 Audio Input
- Converts speech → text using Whisper AI
- Fallback model for reliability
- Outputs:
  - Transcribed text
  - Audio playback

---

# 🧠 AI Models Used

- Image Captioning → Salesforce BLIP
- Speech-to-Text → OpenAI Whisper (tiny/base)
- OCR → Tesseract OCR
- Text-to-Speech → gTTS

---

# 🛠️ Tech Stack

- Python
- Streamlit
- Transformers (Hugging Face)
- PyTorch
- PIL (Pillow)
- gTTS
- Tesseract OCR

---

# 📦 Installation

## 1. Clone repo
```bash
git clone https://github.com/your-username/multimodal-ai-app.git
cd multimodal-ai-app
