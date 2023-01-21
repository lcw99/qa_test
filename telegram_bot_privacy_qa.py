from functools import wraps
import os
from threading import Timer   

from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
from telegram import (ChatAction)

from gpt_index_query_privacy import query, clear_chat_history
from langchain_conversation import build_converation_chain

updater = Updater(os.environ['TELEGRAM_PRIVACY_QA_BOT_TOKEN'], use_context=True)

prompt_template_default = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Summary of conversation:
{history}
Current conversation:
{chat_history_lines}
Human: {input}
AI:"""

prompt_template_doctor = """Here is a friendly conversation between the patient and Patient Navigator.
Patient Navigator listens to the patient's symptoms and, as kindly as possible, explains which doctor the patient should see.
If Patient Navigator doesn't know the answer to a question, honestly say I don't know.

Summary of conversation:
{history}
Current conversation:
{chat_history_lines}
Patient: {input}
Patient Navigator:"""

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
        /couple - 연애상담
		/therapist - 심리상담
        /privacy_qa - 개인정보 FAQ
      """)

def help(update: Update, context: CallbackContext):
	update.message.reply_text("""가능한 상담 :-
        /couple - 연애상담
        /therapist - 심리상담
        /privacy_qa - 개인정보 FAQ
""")


def couple_counselor(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "couple"  # couple, privacy, therapist, doctor
    clear_chat_history(context)
    update.message.reply_text("연애/부부 상담 모드로 전환 되었습니다.")

def therapist(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "therapist"  
    clear_chat_history(context)
    update.message.reply_text("심리상담 모드로 전환 되었습니다.")


def privacy_qa(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "privacy"  
    clear_chat_history(context)
    update.message.reply_text("개인정보 FAQ 모드로 전환 되었습니다.")

def doctor(update: Update, context: CallbackContext):
    context.user_data["councelor_type"] = "doctor"  
    clear_chat_history(context)
    update.message.reply_text("의사 모드로 전환 되었습니다..")

    context.user_data["conversation_doctor"] = build_converation_chain(prompt_template_doctor)
    
def clear_chat_history_handler(update: Update, context: CallbackContext):
    clear_chat_history(context)
    update.message.reply_text("채팅 히스토리가 삭제 되었습니다.")

def send_typing(context, chat_id):
    context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
def unknown(update: Update, context: CallbackContext):
    context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=ChatAction.TYPING)
    t = Timer(8, send_typing, [context, update.effective_message.chat_id])  
    t.start()  
    
    if "councelor_type" not in context.user_data.keys():
        context.user_data["councelor_type"] = "couple"
        update.message.reply_text("연애/부부 상담 모드입니다. 가능한 모드를 보려면 /help 를 치세요.")
        
    q = update.message.text
    q = q.strip()
    # if not q.endswith("?"):
    #     q = q + "?"
    a = query(q, context.user_data["councelor_type"], context)
    
    t.cancel()
    update.message.reply_text(a)


def unknown_text(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Sorry I can't recognize you , you said '%s'" % update.message.text)


updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('help', help))
updater.dispatcher.add_handler(CommandHandler('clear', clear_chat_history_handler))

updater.dispatcher.add_handler(CommandHandler('therapist', therapist))
updater.dispatcher.add_handler(CommandHandler('privacy_qa', privacy_qa))
updater.dispatcher.add_handler(CommandHandler('couple', couple_counselor))
updater.dispatcher.add_handler(CommandHandler('doctor', doctor))

updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown))
updater.dispatcher.add_handler(MessageHandler(
	Filters.command, unknown)) # Filters out unknown commands

# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))

updater.start_polling()
