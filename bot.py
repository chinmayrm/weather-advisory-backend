import logging
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import requests

TOKEN = "7898044134:AAEddtxNHs1gGPiYDIF3hDaTzhWSL6_XRZ0"
BACKEND_URL = "https://weather-advisory-backend.onrender.com/weather"

logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Agri Weather Bot!\n\n"
        "Use `/weather Gadag tomato` to get crop advisory 🌾",
    )

async def get_weather(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 2:
            await update.message.reply_text("❗ Usage: /weather <City> <Crop>")
            return

        city, crop = context.args
        response = requests.get(f"{BACKEND_URL}?city={city}&crop={crop}")
        data = response.json()

        if data["error"]:
            await update.message.reply_text(f"⚠️ Error: {data['error']}")
            return

        message = (
            f"🌾 *Agri Weather Advisory*\n"
            f"*City:* {data['city']}\n"
            f"*Crop:* {data['crop'].capitalize()}\n"
            f"*Condition:* {data['condition']}\n"
            f"*Temperature:* {data['temperature']}°C\n"
            f"*Humidity:* {data['humidity']}%\n\n"
            f"*Advisory:* {data['advisory']}"
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text("❌ Failed to fetch advisory.")

if __name__ == "__main__":
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("weather", get_weather))

    print("✅ Telegram Bot running...")
    app.run_polling()
