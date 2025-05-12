# Deepseek langchain example
# This code demonstrates how to use the Deepseek model with Langchain to generate code and tests based on a given task and programming language.

from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate  
from langchain_core.output_parsers import StrOutputParser

import argparse

load_dotenv()

parser = argparse.ArgumentParser();
parser.add_argument("--task", default="implement a function to perform radix sort on an array of integers", type=str, help="Task to be performed")
parser.add_argument("--language", default="javascript", type=str, help="Programming language to be used")
args = parser.parse_args()

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0)

code_prompt = PromptTemplate(
    input_variables=["language", "task"],
    template="write a very short {language} function that will {task}"
)

code_prompt2 = PromptTemplate(
    input_variables=["language", "code"],
    template="write a test for the following {language} code:\n {code}"
)

chain = code_prompt | llm | StrOutputParser()
chain2 = code_prompt2 | llm | StrOutputParser()

# Execute the first chain
result = chain.invoke({
    "language": args.language,
    "task": args.task
})
print("Code:"+ result)

# Execute the second chain
result2 = chain2.invoke({
    "language": args.language,
    "code": result
})

print("Test:"+ result2) 