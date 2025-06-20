from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes

# Load model and processor
processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")

# Optionally, define medical descriptions (add more if known)
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

# Format response message
def format_prediction(label, confidence):
    desc = MEDICAL_DESCRIPTIONS.get(label, "No medical description available.")
    response = (
        f"🧾 *Diagnosis Result*\n"
        f"• *Condition:* {label}\n"
        f"• *Confidence:* {confidence:.2f}%\n"
        f"• *Description:* {desc}"
    )
    return response

# Handle incoming image
async def classify_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.photo:
        photo = await update.message.photo[-1].get_file()
        image_path = f"{update.message.message_id}.jpg"
        await photo.download_to_drive(image_path)

        try:
            # Process image
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)[0]

            # Get top prediction
            predicted_idx = torch.argmax(probs).item()
            label = model.config.id2label[predicted_idx]
            confidence = probs[predicted_idx].item() * 100

            response = format_prediction(label, confidence)

            # Reply professionally
            await update.message.reply_text(
                response,
                parse_mode="Markdown"
            )

        except Exception as e:
            await update.message.reply_text("⚠️ An error occurred while processing the image. Please try again.")
            print("Error:", e)

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
    else:
        await update.message.reply_text("📎 Please send a clear photo of a skin lesion for analysis.")

# Start the bot
async def main():
    bot_token = "7517844216:AAHXuMFID6ojqdmbJu_s6W8yqPZr176HyK8"  # Replace this
    app = ApplicationBuilder().token(bot_token).build()

    app.add_handler(MessageHandler(filters.PHOTO, classify_image))

    print("🤖 Skin Diagnosis Bot is live.")
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
