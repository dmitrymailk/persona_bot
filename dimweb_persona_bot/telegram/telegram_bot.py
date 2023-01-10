import logging
import random
from typing import TypedDict

from dimweb_persona_bot.inference.seq2seq_bots import DialogBotV3


from telegram import (
    Update,
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import pandas as pd

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DIALOG = range(1)

path = "./models/2uuglxhm/checkpoint-1260000"
model = AutoModelForSeq2SeqLM.from_pretrained(path)
device = "cuda"
model.to(device)
model.eval()
model.half()
tokenizer = AutoTokenizer.from_pretrained(
    path,
    src_lang="ru_RU",
    tgt_lang="ru_RU",
)

ru_bot = DialogBotV3(
    model=model,
    tokenizer=tokenizer,
    history=[],
    debug_status=1,
    device=device,
    max_pairs=0,
)


SUPER_SIMPLE_DATABASE = {}

reply_markup = ReplyKeyboardMarkup(
    [["/start", "/stop"]],
    resize_keyboard=True,
)


class QueueWaiter(TypedDict):
    """A queue waiter."""

    update_object: Update
    user_username: str


class MessageQueue:
    def __init__(
        self,
    ):
        self.queue = []

    def add(self, item: QueueWaiter):
        self.queue.append(item)

    async def availability_check(self):
        """
        бесконечно отвечаем пока в очереди есть запросы.
        это сделано чтобы на видюхе память не кончилась.
        """
        if len(self.queue) > 0:
            waiter = self.queue.pop(0)
            message = waiter["update_object"].message.text
            user_username = waiter["user_username"]

            history = SUPER_SIMPLE_DATABASE[user_username]["history"]

            history.append(message)

            bot2 = DialogBotV3(
                model=model,
                tokenizer=tokenizer,
                history=history,
                max_pairs=0,
                debug_status=1,
            )
            bot_response = bot2.next_response()

            update_object = waiter["update_object"]
            history.append(bot_response)

            if len(history) > 3:
                if history[-3] == history[-1]:
                    history = []

            SUPER_SIMPLE_DATABASE[user_username]["history"] = history

            logger.info("Bot response %s : %s", message, bot_response)

            await update_object.message.reply_text(
                bot_response,
                reply_markup=reply_markup,
            )

            await self.availability_check()


message_queue = MessageQueue()


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:

    await update.message.reply_text(
        "Бот любит когда ты начинаешь писать первым :) Чтобы начать диалог напиши ему привет или что-то типа того."
        "Чтобы закончить диалог напиши /stop",
    )

    user = update.message.from_user
    user_username = user.username

    if user_username not in SUPER_SIMPLE_DATABASE:
        SUPER_SIMPLE_DATABASE[user_username] = {
            "history": [],
        }

    return DIALOG


async def dialog(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> int:
    user = update.message.from_user
    user_username = user.username
    user_text = update.message.text

    if "/stop" in user_text:
        return await stop(update, context)

    logger.info("Message from %s %s: %s", user_username, user.first_name, user_text)

    message_queue.add(
        QueueWaiter(
            update_object=update,
            user_username=user_username,
        )
    )

    await message_queue.availability_check()

    return DIALOG


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    username = None

    if update.message is not None:
        username = update.message.from_user.username
    else:
        username = update.effective_user.username

    if username in SUPER_SIMPLE_DATABASE:
        del SUPER_SIMPLE_DATABASE[username]
    if update.message is not None:
        await update.message.reply_text(
            "До новых встреч!",
        )
    else:
        await update.effective_message.reply_text(
            "До новых встреч!",
        )

    return ConversationHandler.END


async def button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Parses the CallbackQuery and updates the message text."""
    query = update.callback_query
    update.effective_user.username

    # CallbackQueries need to be answered, even if no notification to the user is needed
    # Some clients may have trouble otherwise. See https://core.telegram.org/bots/api#callbackquery
    await query.answer()
    if query.data == "stop":
        await stop(update, context)


def main() -> None:
    """Run the bot."""
    # Create the Application and pass it your bot's token.
    TOKEN = open("./token").read()
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            DIALOG: [MessageHandler(filters.TEXT, dialog)],
        },
        fallbacks=[CommandHandler("stop", stop)],
    )

    application.add_handler(conv_handler)
    application.add_handler(CallbackQueryHandler(button))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
