# Retrieval chain from latihan3.py

from langchain_ollama import OllamaEmbeddings
#from langchain_community.vectorstores import Chroma 
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv


import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0
)

embedding = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434", client_kwargs={"timeout": 600})

db = Chroma(
    embedding_function=embedding,
    persist_directory="./emb"
)

retriever = db.as_retriever()

# Define the prompt for the retrieval chain
# inspired from https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat 
prompt = ChatPromptTemplate(
    input_variables=["input"],
    messages=[
        SystemMessagePromptTemplate.from_template("Answer any use questions based solely on the context : <context>{context}</context>"),
        HumanMessagePromptTemplate.from_template("{input}")
    ]
)

combined_docs_chain = create_stuff_documents_chain(
    llm=llm, 
    document_variable_name="context",
    prompt=prompt
    )

retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=combined_docs_chain
)

result = retrieval_chain.invoke({
    "input": "which one have more bacteria, cellphones or toilet handles ?"
})
print(result["answer"])
print("========================================")