"""
Bot Discord — memanggil FastAPI Demo-day.

Usage (hari 2):
  cd Demo-day/day-2-deployment && python3 discord_bot/bot.py
"""
import os
import sys
from pathlib import Path

import discord
import httpx
from dotenv import load_dotenv

# discord_bot → day-2-deployment → Demo-day
DEMO_DAY_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(DEMO_DAY_ROOT / ".env")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
BOT_PREFIX = os.getenv("BOT_PREFIX", "!")
MENTION_ONLY = os.getenv("MENTION_ONLY", "false").lower() in ("1", "true", "yes")


def ask_api(message: str, beam_size: int = 3) -> str:
    with httpx.Client(timeout=60.0) as client:
        r = client.post(
            f"{API_URL}/chat",
            json={"message": message, "beam_size": beam_size},
        )
        r.raise_for_status()
        data = r.json()
        return data.get("reply", "")


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Bot masuk sebagai {client.user} (API: {API_URL})")


@client.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    text = message.content.strip()
    if not text:
        return

    triggered = False
    user_text = text

    if text.startswith(BOT_PREFIX + "chat"):
        triggered = True
        user_text = text[len(BOT_PREFIX + "chat") :].strip()
    elif client.user and client.user.mentioned_in(message):
        triggered = True
        user_text = text
        for mention in message.mentions:
            user_text = user_text.replace(mention.mention, "").strip()

    if MENTION_ONLY and not (client.user and client.user.mentioned_in(message)):
        if not text.startswith(BOT_PREFIX + "chat"):
            return

    if not triggered:
        return

    if not user_text:
        await message.channel.send("Kirim pertanyaan setelah `!chat` atau mention bot.")
        return

    async with message.channel.typing():
        try:
            reply = await client.loop.run_in_executor(None, ask_api, user_text)
        except httpx.HTTPError as exc:
            await message.channel.send(f"API error: {exc}")
            return
        except Exception as exc:
            await message.channel.send(f"Gagal memproses: {exc}")
            return

    if not reply:
        reply = "(model tidak mengembalikan jawaban — cek checkpoint dan log API)"
    if len(reply) > 1900:
        reply = reply[:1900] + "…"
    await message.channel.send(reply)


def main():
    if not DISCORD_TOKEN:
        print("Set DISCORD_TOKEN di Demo-day/.env", file=sys.stderr)
        sys.exit(1)
    client.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
