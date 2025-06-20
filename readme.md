# 🧑‍⚕️ Skin Cancer Detector Telegram Bot

A powerful AI-driven Telegram bot for instant skin lesion analysis. Send a photo of a skin lesion, and the bot will classify it using a deep learning model and provide a medical description, confidence score, and safety information.

---

## 🚀 Features
- **Image Classification:** Detects and classifies common skin lesions from photos.
- **Medical Insights:** Returns a brief, non-expert description for each diagnosis.
- **Safety Guidance:** Indicates if the condition is likely cancerous, pre-cancerous, or benign.
- **Fast & Private:** All processing is automated and confidential.

---

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```powershell
   git clone (https://github.com/sayanth-t-m/Telegram_skin_cancer_detector.git)
   cd Telegram_skin_cancer_detector
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure your Telegram Bot Token:**
   - Replace the `bot_token` in `bot.py` with your own Telegram bot token from [BotFather](https://t.me/BotFather).

4. **Run the bot:**
   ```powershell
   python bot.py
   ```

---

## 📸 Usage
- Open Telegram and send a clear photo of a skin lesion to your bot.
- The bot will reply with:
  - The predicted condition
  - Confidence score
  - Medical description
  - Safety guidance (cancerous, pre-cancerous, or benign)

---

## 🧬 Model
- Uses a HuggingFace Transformers model: `Anwarkh1/Skin_Cancer-Image_Classification`
- Inference is performed locally for privacy and speed.

---

## ⚠️ Disclaimer
- This bot is **not a substitute for professional medical advice**.
- Always consult a qualified healthcare provider for diagnosis and treatment.

---

## 🤝 Contributing
Pull requests and suggestions are welcome! Please open an issue first to discuss changes.

---

## 📄 License
This project is for educational and research purposes only.
