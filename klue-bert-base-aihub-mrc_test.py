## Load Transformers library
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict_answer(qa_text_pair):
    # Encoding
    encodings = tokenizer(context, question, 
                      max_length=512, 
                      truncation=True,
                      padding="max_length", 
                      return_token_type_ids=False,
                      return_offsets_mapping=True
                      )
    encodings = {key: torch.tensor([val]).to(device) for key, val in encodings.items()}             

    # Predict
    pred = model(encodings["input_ids"], attention_mask=encodings["attention_mask"])
    start_logits, end_logits = pred.start_logits, pred.end_logits
    token_start_index, token_end_index = start_logits.argmax(dim=-1), end_logits.argmax(dim=-1)
    pred_ids = encodings["input_ids"][0][token_start_index: token_end_index + 1]
    answer_text = tokenizer.decode(pred_ids)

    # Offset
    answer_start_offset = int(encodings['offset_mapping'][0][token_start_index][0][0])
    answer_end_offset = int(encodings['offset_mapping'][0][token_end_index][0][1])
    answer_offset = (answer_start_offset, answer_end_offset)
 
    return {'answer_text':answer_text, 'answer_offset':answer_offset}


## Load fine-tuned MRC model by HuggingFace Model Hub ##
HUGGINGFACE_MODEL_PATH = "bespin-global/klue-bert-base-aihub-mrc"
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
model = AutoModelForQuestionAnswering.from_pretrained(HUGGINGFACE_MODEL_PATH).to(device)


## Predict ## 
context = '''
재물운은 따로 수익을 챙기는 것보다는 일 적인 부분에서 가외 수익이 발생하는 시기입니다. 수입에 기복이 있으신 분은 이번 달은 좋은 결과가 기대됩니다. 직장인들도 부수입이 생기는 시기이니 비자금이 따로 마련되겠군요. 재테크를 하기에도 나쁜 달은 아닙니다. 오히려 좋은 달에 속하지요.  애정운은 아직 확신이 서지 않는 연인들은 이번 달이 좋은 기회가 되겠군요. 마음을 숨기는 것이 능사는 아닙니다. 솔직하게 표현하여 서로의 사랑을 더욱 확실하게 다지기 바랍니다. 자신이 속한 곳이 있다면 그 모임에서 두각을 보이는 달입니다. 그 또한 자신의 매력이 됨을 잊지 말아야 합니다. 솔로들은 이러한 모임을 통해서 새로운 인연을 찾는 것이 좋습니다. 건강운은 무리를 해도 좋을 정도로 건강의 리듬이 좋은 달은 아닙니다. 그러나 평소보다는 회복세나 혹은 피로를 느끼는 강도가 훨씬 덜할 것입니다. 직접적인 신체의 리듬보다는 감성과 일적인 부분의 조화가 적절하게 이루어져 피곤함보다는 편안함을 먼저 느끼게 됩니다. 이런 달은 평소보다 회복의 속도가 빠름으로 술을 마시거나 심한 산행을 하시는 것보다는 가벼운 운동이나 사우나를 통해서 몸을 푸시고 묵은 피로를 씻어내도록 하는 것이 좋습니다. 
'''
question = "애정운은 어때?"

qa_text_pair = {'context':context, 'question':question}
result = predict_answer(qa_text_pair)
print('Answer Text: ', result['answer_text'])  # 기존 M1 대비 CPU가 최대 18 %, GPU가 최대 35 % 향상
print('Answer Offset: ', result['answer_offset'])  # (410, 446)
