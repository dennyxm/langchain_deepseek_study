# Deepseek Chat LLM
# This code demonstrates how to use the Deepseek Chat LLM with Langchain to interactively generate responses based on user input.

from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.chat_message_histories.file import FileChatMessageHistory

from pydantic import BaseModel, Field

from dotenv import load_dotenv

import uuid

load_dotenv()

class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: list[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    # if session_id not in store:
    #     store[session_id] = InMemoryHistory()
    if session_id not in store:
        store[session_id] = FileChatMessageHistory(
            "./chat_history.json",
        )
    return store[session_id]

# Initialize the ChatDeepSeek model
chat = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0
)

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        SystemMessagePromptTemplate.from_template("You are a helpful assistant. You will be given a task and you will respond with the result. You can also ask clarifying questions if needed."),
        MessagesPlaceholder(variable_name="history"), 
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = prompt | chat | StrOutputParser()

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_by_session_id,
    input_messages_key="content",
    history_messages_key="history"
)

# Create a unique session ID for the chat
session_id = str(uuid.uuid4())

while True:
    try:
        # Prompt the user for input
        user_input = input(">> ")

        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

        result = chain_with_history.invoke({
            "content": user_input
            },config={ "configurable": { "session_id": session_id } })
         
        print(result)

    except ValueError:
        print("IError: Please provide a valid input.")