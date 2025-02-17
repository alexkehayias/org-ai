# About

Utilities for bringing AI to org-mode.

UPDATE: [I've since replaced this project with indexer](https://github.com/alexkehayias/indexer) and will be archiving this repo.

![org-ai-shell](https://github.com/alexkehayias/org-ai/assets/627790/3c00cac3-4208-4e10-9b15-e83b1439695f)

## Chat with your notes

Use OpenAI GPT models to have a conversation from the contents of your notes.

### Installation

```
source .venv/bin/activate
pip install wheel setuptools pip --upgrade
pip install -r requirements.txt
playwright install
```

### Running it

Run the chat bot:

```
python ./src/chat.py
```
