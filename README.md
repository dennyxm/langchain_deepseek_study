# pycode

This project demonstrates how to use the Deepseek language model with Langchain for code generation, interactive chat, document embedding, and retrieval-augmented question answering.

## Files

### latihan1.py

- Generates code and corresponding tests using the Deepseek model via Langchain.
- Accepts a task description and programming language as arguments.
- Produces a short function and a test for it based on the given task.

### latihan2.py

- Provides an interactive chat interface using the Deepseek model with Langchain.
- Maintains chat history for each session.
- Responds to user input and can ask clarifying questions if needed.

### latihan3.py

- Demonstrates document embedding and similarity search using Ollama embeddings and Chroma vector store.
- Loads and splits a text file (`facts.txt`), embeds the chunks, and persists them to disk.
- Performs a similarity search to find the most relevant document chunk for a given query.

### latihan4.py

- Implements a retrieval-augmented generation (RAG) chain using Deepseek for answering questions based on embedded documents.
- Loads persisted embeddings from `latihan3.py` and retrieves relevant context for the LLM to answer user queries.

## Requirements

- Python 3.x
- `langchain`, `langchain_deepseek`, `langchain_ollama`, `python-dotenv`, `pydantic`, and related dependencies

## Usage

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run code generation:
   ```
   python latihan1.py --task "your task" --language "your language"
   ```

3. Start interactive chat:
   ```
   python latihan2.py
   ```

4. Generate and persist embeddings:
   ```
   python latihan3.py
   ```

5. Run retrieval-augmented QA:
   ```
   python latihan4.py
   ```

6. Type `exit` to end the chat session (for interactive chat).

> Make sure to set up your environment variables as needed (e.g., API keys) in a `.env` file.  
> For embedding, ensure Ollama is running locally and the required model is available.