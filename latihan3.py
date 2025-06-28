# adding context with embedding techniques
# using langchain

import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# https://github.com/deepseek-ai/DeepSeek-V3/issues/806
# as of today ( 13 May 2025 ) DeepSeek does not support embedding yet, 
# therefore the workaround is to use ollama embedding 
# https://ollama.com/blog/embedding-models
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma 

# ollama embedding have some issue related to proxy config, therefore need to explicitly mention to not use proxy 
# https://github.com/langchain-ai/langchain/discussions/21263 
import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

load_dotenv()


embed = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434" ,client_kwargs={"timeout": 600})
                
text_splitter = CharacterTextSplitter(
    separator="\n"
)
text_splitter._chunk_size=200
text_splitter._chunk_overlap=0

loader = TextLoader("./facts.txt")
documents = loader.load_and_split(text_splitter=text_splitter)

print("start embedding "+str(len(documents))+" documents")
start_time = time.time()

db = Chroma.from_documents(
    documents=documents,
    embedding=embed,
    persist_directory="./emb"
)

end_time = time.time()
print("finished embedding")
print("elapsed time for 1 embed document: ", end_time - start_time)
print()
print("========================================")

results = db.similarity_search("which one have more bacteria, cellphones or toilet handles ?", k=1)

for result in results:
    print(result.metadata)
    print(result.page_content)
    print("========================================")