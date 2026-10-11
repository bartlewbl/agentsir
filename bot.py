"""Telegram front end for the agent.

Run with: uv run python bot.py
"""

import asyncio
import logging
import os

import httpx
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.error import BadRequest, RetryAfter
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent import Agents, aiter_events, require_model_credentials
from models import (
    PROVIDERS,
    ModelChoice,
    default_choice,
    filter_models,
    list_models,
    missing_key,
    parse_choice,
    parse_listing,
    provider_default,
)
from persona import apply_persona_command
from trace import format_status

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("agentsir")

TELEGRAM_LIMIT = 4000
ALERT_LIMIT = 200
PICKER_PAGE_SIZE = 8
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
        "While I'm working you can see each step. "
        "/debug keeps those steps after the answer. "
        "/models lists the models I can run on, and /model or /provider picks one. "
        "/reset starts a fresh conversation. "
        "Things I remember about you stay after a reset. "
        "/persona names me or sets how I talk. "
        "/persona bland makes me a plain agent again.",
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    context.user_data["generation"] = context.user_data.get("generation", 0) + 1
    await reply_text(update, "Started a new conversation. Saved memories and your persona stay.")


async def persona(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    await reply_text(
        update,
        apply_persona_command(str(update.effective_user.id), list(context.args or [])),
    )


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    enabled = not context.user_data.get("debug", False)
    context.user_data["debug"] = enabled
    if enabled:
        await reply_text(
            update,
            "Technical mode is on. Thoughts and calls stay in the chat after each answer. "
            "Send /debug again to turn it off.",
        )
        return
    await reply_text(
        update,
        "Technical mode is off. Thoughts and calls disappear once the answer is ready.",
    )


def current_model(context: ContextTypes.DEFAULT_TYPE) -> ModelChoice:
    return context.user_data.get("model") or default_choice()


def switch_model(current: ModelChoice, choice: ModelChoice) -> tuple[ModelChoice, str]:
    """The model to use after asking for `choice`, and the reply to send."""
    key = missing_key(choice)
    if key:
        return current, f"{choice.provider} isn't set up. Add {key} to .env and restart the bot."
    return choice, f"Now using {choice}. The conversation continues."


def choose_model(current: ModelChoice, args: str) -> tuple[ModelChoice, str]:
    """The model to use after a /model command, and the reply to send."""
    try:
        choice = parse_choice(args, current)
    except ValueError as exc:
        return current, str(exc)
    return switch_model(current, choice)


def picker_view(picker: dict, current: ModelChoice) -> tuple[str, InlineKeyboardMarkup]:
    """The text and buttons for one page of the model list."""
    provider, search, models = picker["provider"], picker["search"], picker["models"]
    pages = max(1, -(-len(models) // PICKER_PAGE_SIZE))
    page = min(max(picker["page"], 0), pages - 1)
    first = page * PICKER_PAGE_SIZE
    shown = models[first : first + PICKER_PAGE_SIZE]

    matching = f' matching "{search}"' if search else ""
    lines = [f"Model: {current}", ""]
    if shown:
        lines.append(
            f"{provider} models{matching}, {first + 1}–{first + len(shown)} of {len(models)}. "
            "Tap one to use it."
        )
    else:
        lines.append(f"No {provider} models{matching}.")
    key = missing_key(ModelChoice(provider, ""))
    if key:
        lines.append(f"{provider} needs {key} in .env before you can pick one.")
    lines.append("")
    lines.append(
        "/models <words> narrows the list, for example /models openrouter claude. "
        "/model <id> takes any id, listed or not."
    )

    rows = []
    for index, name in enumerate(shown, start=first):
        mark = "✓ " if ModelChoice(provider, name) == current else ""
        rows.append([InlineKeyboardButton(mark + name, callback_data=f"pick:m:{index}")])
    if pages > 1:
        nav = []
        if page > 0:
            nav.append(InlineKeyboardButton("◀", callback_data=f"pick:p:{page - 1}"))
        nav.append(InlineKeyboardButton(f"{page + 1}/{pages}", callback_data="pick:noop"))
        if page < pages - 1:
            nav.append(InlineKeyboardButton("▶", callback_data=f"pick:p:{page + 1}"))
        rows.append(nav)
    others = [name for name in PROVIDERS if name != provider]
    if others:
        rows.append(
            [InlineKeyboardButton(f"Show {name} models", callback_data=f"pick:v:{name}") for name in others]
        )
    return "\n".join(lines), InlineKeyboardMarkup(rows)


async def load_picker(provider: str, search: str) -> tuple[dict | None, str]:
    """A fresh model list for a provider, or the reason it couldn't be loaded."""
    try:
        models = await list_models(provider)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        logger.warning("could not list %s models: %s", provider, exc)
        key = missing_key(ModelChoice(provider, ""))
        reason = f"it needs {key} in .env" if key else "the request failed"
        return None, (
            f"I couldn't load the {provider} model list: {reason}. "
            f"Browse it at {PROVIDERS[provider].models_url} and use /model {provider} <id>."
        )
    picker = {
        "provider": provider,
        "search": search,
        "models": filter_models(models, search),
        "page": 0,
    }
    return picker, ""


async def send_picker(update: Update, context: ContextTypes.DEFAULT_TYPE, args: str) -> None:
    current = current_model(context)
    provider, search = parse_listing(args, current)
    picker, problem = await load_picker(provider, search)
    if picker is None:
        await reply_text(update, problem)
        return
    text, buttons = picker_view(picker, current)
    sent = await update.effective_message.reply_text(text, reply_markup=buttons)
    picker["message_id"] = sent.message_id
    context.user_data["picker"] = picker


async def models_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    await send_picker(update, context, " ".join(context.args))


async def model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    if not context.args:
        await send_picker(update, context, "")
        return
    choice, reply = choose_model(current_model(context), " ".join(context.args))
    context.user_data["model"] = choice
    await reply_text(update, reply)


async def provider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if await reject_unwanted(update, context):
        return
    if len(context.args) != 1:
        await send_picker(update, context, "")
        return
    name = context.args[0].lower()
    if name not in PROVIDERS:
        await reply_text(update, f"Unknown provider {name}. Choose one of: {', '.join(PROVIDERS)}.")
        return
    choice, reply = switch_model(current_model(context), provider_default(name))
    context.user_data["model"] = choice
    await reply_text(update, reply)


async def on_pick(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """A tap on a button under the model list."""
    query = update.callback_query
    allowed = context.application.bot_data["allowed_ids"]
    if allowed is not None and query.from_user.id not in allowed:
        await query.answer("This bot is private.")
        return
    picker = context.user_data.get("picker")
    if picker is None or query.message is None or picker.get("message_id") != query.message.message_id:
        await query.answer("This list is out of date. Send /models for a new one.")
        return

    action, _, value = query.data.removeprefix("pick:").partition(":")
    current = current_model(context)
    if action == "m" and value.isdigit() and int(value) < len(picker["models"]):
        target = ModelChoice(picker["provider"], picker["models"][int(value)])
        choice, reply = switch_model(current, target)
        if choice != target:
            await query.answer(reply[:ALERT_LIMIT], show_alert=True)
            return
        context.user_data["model"] = choice
        del context.user_data["picker"]
        await query.answer()
        await edit_text(query.message, reply)
        return
    if action == "p" and value.isdigit():
        picker["page"] = int(value)
    elif action == "v" and value in PROVIDERS:
        fresh, problem = await load_picker(value, "")
        if fresh is None:
            await query.answer(problem[:ALERT_LIMIT], show_alert=True)
            return
        fresh["message_id"] = picker["message_id"]
        picker = context.user_data["picker"] = fresh
    else:
        await query.answer()
        return

    await query.answer()
    text, buttons = picker_view(picker, current)
    try:
        await query.message.edit_text(text, reply_markup=buttons)
    except BadRequest as exc:
        if "not modified" not in str(exc).lower():
            logger.warning("could not update model list: %s", exc)


def trace_after_answer(debug_mode: bool, steps: list[dict]) -> str:
    """The step list to leave in the chat, empty unless technical mode is on."""
    if not debug_mode:
        return ""
    return format_status(steps)


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
    choice = current_model(context)
    try:
        async for event in aiter_events(
            context.application.bot_data["agents"].get(choice),
            question,
            thread_id=thread_id(update, context),
            user_id=str(update.effective_user.id),
        ):
            if event["type"] == "answer":
                answer = event["text"]
                continue
            steps.append(event)
            rendered = format_status(steps)
            if rendered:
                await edit_text(status, fit_message(rendered))
    except Exception as exc:
        logger.exception("agent failed on %s", choice)
        answer = (
            f"Something went wrong while answering with {choice}: {str(exc)[:300]}\n\n"
            "Try again, or pick another model with /model."
        )

    answer = answer.strip() or "I don't have a reply for that."
    trace = trace_after_answer(bool(context.user_data.get("debug")), steps)
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
    await reply_text(
        update,
        "I don't know that command. "
        "Try /start, /reset, /debug, /models, /model, /provider, or /persona.",
    )


def main() -> None:
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "Set TELEGRAM_BOT_TOKEN. Create a bot with @BotFather and put the token in .env."
        )
    require_model_credentials()

    application = Application.builder().token(token).build()
    application.bot_data["agents"] = Agents()
    application.bot_data["allowed_ids"] = allowed_user_ids()
    application.add_handler(CommandHandler("start", start, filters=PRIVATE))
    application.add_handler(CommandHandler("reset", reset, filters=PRIVATE))
    application.add_handler(CommandHandler("debug", debug, filters=PRIVATE))
    application.add_handler(CommandHandler("model", model, filters=PRIVATE))
    application.add_handler(CommandHandler("models", models_list, filters=PRIVATE))
    application.add_handler(CallbackQueryHandler(on_pick, pattern=r"^pick:"))
    application.add_handler(CommandHandler("provider", provider, filters=PRIVATE))
    application.add_handler(CommandHandler("persona", persona, filters=PRIVATE))
    application.add_handler(MessageHandler(PRIVATE & filters.TEXT & ~filters.COMMAND, on_text))
    application.add_handler(MessageHandler(PRIVATE & filters.COMMAND, unknown_command))
    application.add_handler(MessageHandler(PRIVATE & ~filters.TEXT & ~filters.COMMAND, on_other))

    logger.info("telegram bot polling")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
