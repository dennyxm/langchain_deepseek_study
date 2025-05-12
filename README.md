# pycode

This project demonstrates how to use the Deepseek language model with Langchain for code generation and interactive chat.

## Files

### latihan1.py

- Generates code and corresponding tests using the Deepseek model via Langchain.
- Accepts a task description and programming language as arguments.
- Produces a short function and a test for it based on the given task.

### latihan2.py

- Provides an interactive chat interface using the Deepseek model with Langchain.
- Maintains chat history for each session.
- Responds to user input and can ask clarifying questions if needed.

## Requirements

- Python 3.x
- `langchain`, `langchain_deepseek`, `python-dotenv`, `pydantic`, and related dependencies

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

4. Type `exit` to end the chat session.

> Make sure to set up your environment variables as needed (e.g., API keys) in a `.env` file.
