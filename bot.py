"""Telegram front end for the agent.

Run with: uv run python bot.py
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent import aiter_events, build_agent, require_model_credentials
from trace import format_trace

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("agentsir")

TELEGRAM_LIMIT = 4000
PRIVATE = filters.ChatType.PRIVATE


def allowed_user_ids() -> set[int] | None:
    raw = os.environ.get("TELEGRAM_ALLOWED_USER_IDS", "").strip()
    if not raw:
        return None
    try:
        return {int(part.strip()) for part in raw.split(",") if part.strip()}
    except ValueError as exc:
        raise SystemExit(
            "TELEGRAM_ALLOWED_USER_IDS must be a comma-separated list of numeric Telegram user ids."
        ) from exc


def thread_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    user_id = update.effective_user.id
    generation = context.user_data.get("generation", 0)
    return f"{user_id}:{generation}"


async def reject_unwanted(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    user = update.effective_user
    if user is None or update.effective_message is None:
        return True
    allowed = context.application.bot_data["allowed_ids"]
    if allowed is not None and user.id not in allowed:
        await update.effective_message.reply_text("This bot is private.")
        return True
    return False


def text_chunks(text: str) -> list[str]:
    text = text.strip() or "I don't have a reply for that."
    if len(text) <= TELEGRAM_LIMIT:
        return [text]
    parts = []
    while text:
        if len(text) <= TELEGRAM_LIMIT:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, TELEGRAM_LIMIT)
        if cut < TELEGRAM_LIMIT // 2:
            cut = TELEGRAM_LIMIT
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip("\n")
    return [part for part in parts if part]


async def reply_text(update: Update, text: str) -> None:
    for part in text_chunks(text):
        await update.effective_message.reply_text(part)


def fit_message(text: str) -> str:
    text = text.strip()
    if len(text) <= TELEGRAM_LIMIT:
        return text
    return "…\n" + text[-(TELEGRAM_LIMIT - 2) :].lstrip()


async def edit_text(message, text: str) -> bool:
    for _ in range(2):
        try:
            await message.edit_text(text)
            return True
        except RetryAfter as exc:
            await asyncio.sleep(exc.retry_after + 0.1)
        except BadRequest as exc:
            if "not modified" in str(exc).lower():
                return True
            logger.warning("could not edit message: %s", exc)
            return False
    return False


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    await reply_text(
        update,
        "Send me a message and I'll answer. "
        "When I use a tool, that step shows up first. "
        "/reset starts a fresh conversation.",
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    context.user_data["generation"] = context.user_data.get("generation", 0) + 1
    await reply_text(update, "Started a new conversation.")


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    question = update.effective_message.text
    if not question or not question.strip():
        return

    logger.info("message from user %s", update.effective_user.id)
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )
    status = await update.effective_message.reply_text("Thinking…")
    steps = []
    answer = ""
    try:
        async for event in aiter_events(
            context.application.bot_data["agent"],
            question,
            thread_id=thread_id(update, context),
        ):
            if event["type"] == "answer":
                answer = event["text"]
                continue
            steps.append(event)
            rendered = format_trace(steps)
            if rendered:
                await edit_text(status, fit_message(rendered))
    except Exception:
        logger.exception("agent failed")
        answer = "Something went wrong while answering. Try again."

    answer = answer.strip() or "I don't have a reply for that."
    trace = format_trace(steps)
    if trace:
        await edit_text(status, fit_message(trace))
        await reply_text(update, answer)
        return

    chunks = text_chunks(answer)
    if not await edit_text(status, chunks[0]):
        await reply_text(update, answer)
        return
    for part in chunks[1:]:
        await update.effective_message.reply_text(part)


async def on_other(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    await reply_text(update, "Send a text message.")


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    await reply_text(update, "I don't know that command. Try /start or /reset.")


def main() -> None:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "Set TELEGRAM_BOT_TOKEN. Create a bot with @BotFather and put the token in .env."
        )
    require_model_credentials()

    application = Application.builder().token(token).build()
    application.bot_data["agent"] = build_agent()
    application.bot_data["allowed_ids"] = allowed_user_ids()
    application.add_handler(CommandHandler("start", start, filters=PRIVATE))
    application.add_handler(CommandHandler("reset", reset, filters=PRIVATE))
    application.add_handler(MessageHandler(PRIVATE & filters.TEXT & ~filters.COMMAND, on_text))
    application.add_handler(MessageHandler(PRIVATE & filters.COMMAND, unknown_command))
    application.add_handler(MessageHandler(PRIVATE & ~filters.TEXT & ~filters.COMMAND, on_other))

    logger.info("telegram bot polling")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
