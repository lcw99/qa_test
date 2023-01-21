from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory, ConversationSummaryMemory, CombinedMemory

def build_converation_chain(prompt_template):
    conv_memory = ConversationBufferMemory(
        memory_key="chat_history_lines",
        input_key="input"
    )

    summary_memory = ConversationSummaryMemory(llm=OpenAI(), input_key="input")
    # Combined
    memory = CombinedMemory(memories=[conv_memory, summary_memory])
    PROMPT = PromptTemplate(
        input_variables=["history", "input", "chat_history_lines"], template=prompt_template
    )
    llm = OpenAI(temperature=0)
    conversation = ConversationChain(
        llm=llm, 
        verbose=True, 
        memory=memory,
        prompt=PROMPT
    )
    return conversation