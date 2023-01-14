from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
from telegram import (ChatAction)

from gpt_index_query_privacy import query
from functools import wraps
import os
from threading import Timer   

updater = Updater(os.environ['TELEGRAM_PRIVACY_QA_BOT_TOKEN'], use_context=True)

therapist_mode = True

def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    def command_func(update, context, *args, **kwargs):
        context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
        return func(update, context,  *args, **kwargs)

    return command_func

def start(update: Update, context: CallbackContext):
	update.message.reply_text(
		"""
        인공지능 상담 봇입니다.
		/therapist - 심리상담
        /privacy_qa - 개인정보 FAQ
      """)

def help(update: Update, context: CallbackContext):
	update.message.reply_text("""Available Commands :-
		/therapist - 심리상담
        /privacy_qa - 개인정보 FAQ
""")


def gmail_url(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Your gmail link here (I am not\
		giving mine one for security reasons)")


def therapist(update: Update, context: CallbackContext):
    global therapist_mode
    
    therapist_mode = True
    update.message.reply_text("심리상담 모드로 전환 되었습니다.")


def privacy_qa(update: Update, context: CallbackContext):
    global therapist_mode

    therapist_mode = False
    update.message.reply_text("개인정보 FAQ 모드로 전환 되었습니다.")


def geeks_url(update: Update, context: CallbackContext):
	update.message.reply_text(
		"GeeksforGeeks URL => https://www.geeksforgeeks.org/")

def send_typing(context, chat_id):
    context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
def unknown(update: Update, context: CallbackContext):
    global therapist_mode

    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
    t = Timer(8, send_typing, [context, update.effective_message.chat_id])  
    t.start()  
        
    q = update.message.text
    q = q.strip()
    # if not q.endswith("?"):
    #     q = q + "?"
    type = "privacy"
    if therapist_mode:
        type = "therapist"
    a = query(q, type)
    
    t.cancel()
    update.message.reply_text(a)


def unknown_text(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Sorry I can't recognize you , you said '%s'" % update.message.text)


updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('therapist', therapist))
updater.dispatcher.add_handler(CommandHandler('help', help))
updater.dispatcher.add_handler(CommandHandler('privacy_qa', privacy_qa))
updater.dispatcher.add_handler(CommandHandler('gmail', gmail_url))
updater.dispatcher.add_handler(CommandHandler('geeks', geeks_url))
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))
updater.dispatcher.add_handler(MessageHandler(
	Filters.command, unknown)) # Filters out unknown commands

# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))

updater.start_polling()
