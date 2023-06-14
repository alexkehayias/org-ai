# About

Utilities for bringing AI to org-mode.

# Semantic search

Use `txtai` to create an index for searching notes by a similarity query.

## Installation

Install `openjdk`

```
brew install openjdk
sudo ln -sfn $(brew --prefix openjdk)/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk
```

Install `protobuf`

```
brew install protobuf
```

Install python dependencies

```
pip3 install git+https://github.com/neuml/txtai
pip3 install git+https://github.com/neuml/txtai#egg=txtai[pipeline]
```

## Running the example

```
python3 src/run.py
```

# Chat with your notes

Use OpenAI GPT models to have a conversation from the contents of your notes.

## Installation

```
source ./bin/activate
pip3 install -r requirements.txt
```

## Running the example

```
python3 src/chat.py
```
