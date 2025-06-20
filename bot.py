from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os
import re
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")

# Markdown escape helper for Telegram
def escape_markdown(text: str) -> str:
    escape_chars = r'_*\[\]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# Fetch condition info from Gemini
def fetch_condition_info(condition: str) -> (str, str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    prompt = (
        f"Is '{condition}' cancerous or non-cancerous? Harmful or harmless? "
        f"Answer in one line FIRST like this format: "
        f"❗ Harmful - Cancerous OR ✅ Harmless - Non-Cancerous OR ⚠️ Potentially Harmful - Pre-cancerous.\n\n"
        f"Then provide a brief explanation for a non-expert."
        f"What are its signs?."
        f"How to treat?."
        f"What should I do?."
    )

    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        full_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        lines = full_text.strip().split('\n', 1)
        safety_label = lines[0].strip()
        explanation = lines[1].strip() if len(lines) > 1 else "No additional info available."
        return safety_label, explanation
    except Exception as e:
        print(f"[ERROR] Gemini API failed: {e}")
        return "⚠️ Unknown Severity", "Medical information unavailable at this time. Please consult a specialist."

# Format Telegram response
def format_prediction(label: str, confidence: float, safety: str, info: str) -> str:
    label_esc = escape_markdown(label)
    confidence_esc = escape_markdown(f"{confidence:.2f}%")
    safety_esc = escape_markdown(safety)
    info_esc = escape_markdown(info)

    return (
        f"🧾 *Diagnosis Result*\n"
        f"• *Condition:* {label_esc}\n"
        f"• *Confidence:* {confidence_esc}\n"
        f"• *Type:* {safety_esc}\n\n"
        f"*Medical Insight:*\n{info_esc}"
    )

# Handle incoming image
async def classify_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo:
        print(f"[EVENT] Received photo: message_id={update.message.message_id}")
        photo = await update.message.photo[-1].get_file()
        image_path = f"{update.message.message_id}.jpg"
        await photo.download_to_drive(image_path)

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

            predicted_idx = torch.argmax(probs).item()
            label = model.config.id2label[predicted_idx]
            confidence = probs[predicted_idx].item() * 100

            safety, info = fetch_condition_info(label)
            response = format_prediction(label, confidence, safety, info)

            print(f"[EVENT] Diagnosis: {label} ({confidence:.2f}%)")
            await update.message.reply_text(response, parse_mode="MarkdownV2")

        except Exception as e:
            print(f"[ERROR] {e}")
            await update.message.reply_text("⚠️ An error occurred while processing the image. Please try again.")

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
    else:
        await update.message.reply_text("📎 Please send a clear photo of a skin lesion.")

# Start the bot
async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(MessageHandler(filters.PHOTO, classify_image))
    print("🤖 Skin Diagnosis Bot is live.")
    await app.run_polling(close_loop=False)

# Entry point
if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "already running" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.get_event_loop().run_until_complete(main())
        else:
            raise
