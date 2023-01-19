from gpt_index import GPTTreeIndex, SimpleDirectoryReader, LLMPredictor, QueryMode, GPTListIndex, MockLLMPredictor
from IPython.display import Markdown, display
from langchain import OpenAI
import json, os
from deep_translator import GoogleTranslator
from langdetect import detect
import fasttext
import gcld3

model_name = "text-davinci-003"
#model_name = "text-curie-001"

data_folder = "data_therapist"
index_file_name = 'privacy_index_eng.json'

pretrained_lang_model = "fasttext/lid.176.ftz"
fasttext_model = fasttext.load_model(pretrained_lang_model)

detector = gcld3.NNetLanguageIdentifier(min_num_bytes=0, 
                                        max_num_bytes=1000)

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=model_name))
        
def build_index():
    documents = SimpleDirectoryReader(data_folder).load_data()
    index = GPTTreeIndex(documents, num_children=10)
    #index = GPTListIndex(documents)

    index.save_to_disk(index_file_name)

if not os.path.exists(index_file_name):
    build_index()


def query(text, type):
    global index_file_name
    
    if type == "therapist":
        index_file_name = "therapist.json"
    else:
        index_file_name = "privacy_index_eng.json"
    print(f'---- loading index file: {index_file_name}')
    new_index = GPTTreeIndex.load_from_disk(index_file_name, llm_predictor=llm_predictor)

    q = text
    # input_lang = detect(q)
    # input_lang = detector.FindLanguage(text=q).language
    result = fasttext_model.predict(q)
    print(result)
    input_lang = result[0][0].replace('__label__', '')
    print(input_lang, q)
    q = GoogleTranslator(source='auto', target='en').translate(q)
    print(q)
    
    response = new_index.query(q, verbose=True, mode=QueryMode.EMBEDDING)
    print(f'====\n{response.get_formatted_sources()}\n===\n')
    response = GoogleTranslator(source='auto', target=input_lang).translate(response.response)
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
    