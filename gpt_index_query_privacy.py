from gpt_index import GPTTreeIndex, SimpleDirectoryReader, LLMPredictor, QueryMode, GPTListIndex, MockLLMPredictor
from IPython.display import Markdown, display
from langchain import OpenAI
import json, os
from deep_translator import GoogleTranslator
from langdetect import detect
import fasttext
import gcld3
from telegram.ext.callbackcontext import CallbackContext

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory

prompt_template_default = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, it truthfully says it does not know.

Summary of conversation:
{history}
Current conversation:
{chat_history_lines}
Human: {input}
AI:"""

prompt_template_doctor = """The conversation below is between AI counselor and a potential patient. The counselor identifies the patient's condition, symptoms, and current location and writes a report. Once the patient's condition, symptoms, and location are identified, the counselor writes a formal report which include patient's condition, symptoms, and location. Report shound be shown here for future refence.

Current conversation:
{chat_history_lines}
Patient: {input}
Receptionist: """

prompt_template_mbti = """The following is a conversation between an MBTI tester and a customer. MBTI testers try to identify the customer's personality as accurately as possible, and when the customer's personality is identified, they give a detailed explanation to the customer.

Current conversation:
{chat_history_lines}
customer: {input}
tester: """

model_name = "text-davinci-003"
#model_name = "text-curie-001"

data_folder = "couple_counseling_data"
index_file_name = 'couple_counseling_data_list.json'

pretrained_lang_model = "fasttext/lid.176.bin"
fasttext_model = fasttext.load_model(pretrained_lang_model)

detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, 
                                        max_num_bytes=1000)

def build_conversation_chain(input_variables, prompt_template, human_prefix, ai_prefix):
    conv_memory = ConversationBufferMemory(
        memory_key="chat_history_lines",
        input_key="input",
        human_prefix=human_prefix,
        ai_prefix=ai_prefix
    )

    # summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")
    # Combined
    # memory = CombinedMemory(memories=[conv_memory, summary_memory])
    memory = conv_memory
    PROMPT = PromptTemplate(
        input_variables=input_variables, template=prompt_template
    )
    llm = OpenAI(temperature=0.3)
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory,
        prompt=PROMPT
    )
    return conversation

def build_index():
    documents = SimpleDirectoryReader(data_folder).load_data()
    index = GPTTreeIndex(documents, num_children=10)
    #index = GPTListIndex(documents)

    index.save_to_disk(index_file_name)

if not os.path.exists(index_file_name):
    build_index()

def clear_chat_history(context: CallbackContext):
    if "llm_predictor" in context.user_data.keys():
        llm_predictor = context.user_data["llm_predictor"]
        llm_predictor.clear_chat_history()
    if "conversation_chain" in context.user_data.keys():
        conversation = context.user_data["conversation_chain"]
        conversation.memory.clear()
        
def translate_to_english(text):
    q = text
    input_lang1 = detect(q)
    # input_lang = detector.FindLanguage(text=q).language
    result = fasttext_model.predict(q.replace("\n", ""))
    print(result)
    input_lang2 = result[0][0].replace('__label__', '')
    if input_lang1 == input_lang2:
        input_lang = input_lang1
    else:
        input_lang = 'ko'
    print(input_lang, q)
    q = GoogleTranslator(source='auto', target='en').translate(q)
    # q = GoogleTranslator(source='ja', target='en').translate(q)
    print(q)
    return input_lang, q

def naver_request(url, text, source_lang = None, target_lang = None):
    import urllib.request
    encText = urllib.parse.quote(text)
    if source_lang is None:
        data = "query=" + encText
    else:
        data = f"source={source_lang}&target={target_lang}&text=" + encText    
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id","gDewaOS1Yt3Skemibggq")
    request.add_header("X-Naver-Client-Secret", "cQqrQRfHnZ")
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
        response = json.loads(response_body.decode('utf-8'))
        return response

    else:
        print("Error Code:" + rescode)
        return None
        
def translate_papago(text, source_lang, target_lang):
    if source_lang == 'auto':
        lang_res = naver_request("https://openapi.naver.com/v1/papago/detectLangs", text[:10])
        print(lang_res)
        source_lang = lang_res["langCode"]
    response = naver_request("https://openapi.naver.com/v1/papago/n2mt", text, source_lang, target_lang)
    if response is not None:
        response = response["message"]["result"]["translatedText"]
        return source_lang, response
    else:
        return 'ko', '에러'
        
def query(text, type, context: CallbackContext):
    if type in ["doctor", "mbti"]:
        if type == "doctor":
            prompt_temp = prompt_template_doctor
            human_prefix = "Patient"
            ai_prefix = "Counselor"
        elif type == "mbti":
            prompt_temp = prompt_template_mbti
            human_prefix = "Customer"
            ai_prefix = "Tester"
            
        if "conversation_chain" not in context.user_data.keys():
            context.user_data["conversation_chain"] = build_conversation_chain(["input", "chat_history_lines"], prompt_temp, human_prefix, ai_prefix)
        conversation_chain = context.user_data["conversation_chain"]
        input_lang, q = translate_to_english(text)
        #input_lang, q = translate_papago(text, 'auto', 'en')
        response_en = conversation_chain.run(q)
        print(response_en)
        response = GoogleTranslator(source='en', target=input_lang).translate(response_en)
        #input_lang, response = translate_papago(response_en, 'en', input_lang)
        return f'{response_en}\n\n{response}'
        
    if "llm_predictor" not in context.user_data.keys():
        context.user_data["llm_predictor"] = LLMPredictor(llm=OpenAI(temperature=0.5, model_name=model_name), chat_history=3)
    llm_predictor = context.user_data["llm_predictor"]
    
    if type == "therapist":
        index_file_name_local = "therapist.json"
    elif type == "privacy":
        index_file_name_local = "privacy_index_eng.json"
    elif type == "couple":
        index_file_name_local = "couple_counseling_data.json"
        
    print(f'---- loading index file: {index_file_name_local}')
    new_index = GPTTreeIndex.load_from_disk(index_file_name_local, llm_predictor=llm_predictor)
    # new_index = GPTListIndex.load_from_disk(index_file_name_local, llm_predictor=llm_predictor)

    input_lang, q = translate_to_english(text)
    
    response = new_index.query(q, verbose=True, mode=QueryMode.EMBEDDING)
    # response = new_index.query(q, verbose=True, response_mode="tree_summarize")
    # print(f'====\n{response.get_formatted_sources()}\n===\n')
    response = GoogleTranslator(source='en', target=input_lang).translate(response.response)
    if response.startswith("A:"):
        response = response[2:].strip()
    print(response)
    return response


if __name__ == '__main__':
    #q = "What are the procedures to be followed in case of closure?"
    #q = "Who should be designated as the person responsible for personal information?
    #q = "What steps must be taken to transfer the business?"
    #q = "What should be included in the privacy policy?"
    #q = "Is it possible to collect disability grade information?"
    #q = "Trying to outsource customer service. What items must be included in the contract when signing a personal information handling consignment contract with an outsourcing company?"
    #q = "What is the guide board standard and material?"

    #q = "안내판의 규격이나 재질은?"
    #q = "개인정보 교육을 외부 기관에서 수강해도 되나요?"
    #q = "회사 사업자등록정보도 개인정보 인가?"
    #q = "개인정보 수신동의를 몇년마다 받아야 하는가?"
    #q = "아동의 개인정보는 어떻게 수집하는가?"
    #q = "개인정보 관리책임자의 자격은?"
    q = "너무 외로워요. 어떻게 해야 할까요?"
    a = query(q)
    print(a)
    