from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os
import re
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")

# Medical descriptions for conditions
MEDICAL_DESCRIPTIONS = {
    "Basal Cell Carcinoma": "A common type of skin cancer that arises from basal cells. Usually slow-growing and rarely spreads.",
    "Melanoma": "A serious form of skin cancer that begins in cells known as melanocytes. Can spread rapidly if not caught early.",
    "Benign Keratosis": "Non-cancerous skin growth, often appearing as a brown, black, or light tan patch. Typically harmless.",
    "Actinic Keratosis": "A rough, scaly patch on your skin caused by years of sun exposure. Can be a precursor to skin cancer.",
    "Dermatofibroma": "A common benign skin nodule, typically firm and raised. Often harmless.",
    "Vascular Lesion": "An abnormality of blood vessels in the skin. Can be benign or malignant depending on the type.",
    "Squamous Cell Carcinoma": "A type of skin cancer that may appear as a scaly red patch or sore. Can become invasive.",
    "Seborrheic Keratosis": "A benign, often pigmented growth that looks like it's stuck onto the skin. Very common in older adults."
}

# MarkdownV2 escape utility
def escape_markdown(text: str) -> str:
    escape_chars = r'_*\[\]()~`>#+-=|{}.!'
    return re.sub(f'([{re.escape(escape_chars)}])', r'\\\1', text)

# Format diagnosis message
def format_prediction(label: str, confidence: float) -> str:
    desc = MEDICAL_DESCRIPTIONS.get(label, "No medical description available.")
    label_esc = escape_markdown(label)
    desc_esc = escape_markdown(desc)
    confidence_esc = escape_markdown(f"{confidence:.2f}%")

    return (
        f"🧾 *Diagnosis Result*\n"
        f"• *Condition:* {label_esc}\n"
        f"• *Confidence:* {confidence_esc}\n"
        f"• *Description:* {desc_esc}"
    )

# Handle image classification
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

            response = format_prediction(label, confidence)

            print(f"[EVENT] Diagnosis: {label} ({confidence:.2f}%)")
            await update.message.reply_text(response, parse_mode="MarkdownV2")

        except Exception as e:
            print(f"[ERROR] {e}")
            await update.message.reply_text("⚠️ An error occurred while processing the image. Please try again.")

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
    else:
        await update.message.reply_text("📎 Please send a clear photo of a skin lesion for analysis.")

# Start the bot
async def main():
    bot_token = "7517844216:AAHXuMFID6ojqdmbJu_s6W8yqPZr176HyK8"
    app = ApplicationBuilder().token(bot_token).build()
    app.add_handler(MessageHandler(filters.PHOTO, classify_image))

    print("🤖 Skin Diagnosis Bot is live.")
    await app.run_polling(close_loop=False)

# Async-safe runner for various environments
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
