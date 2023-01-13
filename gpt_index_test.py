from gpt_index import GPTTreeIndex, SimpleDirectoryReader, LLMPredictor, QueryMode, GPTListIndex
from IPython.display import Markdown, display
from langchain import OpenAI
import json, os
from deep_translator import GoogleTranslator

index_file_name = 'privacy_index_eng.json'

def build_index():
    documents = SimpleDirectoryReader('data_eng').load_data()
    index = GPTTreeIndex(documents, num_children=10)
    #index = GPTListIndex(documents)

    index.save_to_disk(index_file_name)

if not os.path.exists(index_file_name):
    build_index()

llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
new_index = GPTTreeIndex.load_from_disk(index_file_name, llm_predictor=llm_predictor)

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
q = "아동의 개인정보는 어떻게 수집하는가?"
print(q)
q = GoogleTranslator(source='auto', target='en').translate(q)
print(q)
response = new_index.query(q, verbose=True, mode=QueryMode.EMBEDDING)
print(response)
response = GoogleTranslator(source='auto', target='ko').translate(response.response)
print(response)