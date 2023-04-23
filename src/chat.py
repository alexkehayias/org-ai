import sys
import os
import pickle
import glob
import re

from langchain.chat_models import ChatOpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

TEMPLATE = """You are a helpful personal assistant. Respond clearly and concisely. Given the following data, create a final answer ("FINAL ANSWER") with a list of references ("SOURCES"). Only use references that match the question. If you don't know the answer, just say that you don't know. Don't try to make up an answer. ALWAYS return a "SOURCES" part in your answer.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
PROMPT = PromptTemplate(
    template=TEMPLATE, input_variables=["summaries", "question"]
)


CHAIN = load_qa_with_sources_chain(
    ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo",
        temperature=0,
    ),
    chain_type="stuff",
    prompt=PROMPT,
)


def extract_links_and_replace_text(text):
    regex = r'\[\[id:(.+?)\]\[(.+?)\]\]'
    matches = re.findall(regex, text)
    for match in matches:
        text = text.replace("[[id:{}][{}]]".format(match[0], match[1]), match[1])
    return matches, text


def extract_note(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    id = ''
    title = ''
    links = []
    body = ''
    read_body = False

    for line in lines:
        if line.startswith('#+TITLE:'):
            title = line.replace('#+TITLE: ', '').capitalize().strip()
            continue

        if line.startswith('#+'):
            continue

        if line.startswith(':PROPERTIES:'):
            continue

        if line.startswith(':ID:'):
            id = line.replace(':ID:', '').strip()
            continue

        if line.startswith(':END:'):
            read_body = True
            continue

        if read_body:
            matches, text = extract_links_and_replace_text(line)
            links += matches
            body += text

    return id, title, body.strip(), links


def build_search_index_and_embeddings(path):
    """
    Builds a search index based on vectors of embeddings.
    """
    sources = []
    for filename in glob.glob(f"{path}/*.org"):
        print(f"Indexing {filename}")
        id, title, body, links = extract_note(filename)

        if not body:
            print(f"Skipping note because the body is empty: {filename}")
            continue

        # Skip journal entries
        if "journal" in title.lower():
            continue

        doc = Document(
            page_content=body,
            metadata={
                "id": id,
                "links": links,
                "source": filename,
            },
        )

        sources.append(doc)

    with open("note_search_index.pickle", "wb") as f:
        index = FAISS.from_documents(
            documents=sources,
            embedding=OpenAIEmbeddings(
                openai_api_key=OPENAI_API_KEY,
            ),
        )
        pickle.dump(index, f)


def search_index():
    with open("note_search_index.pickle", "rb") as f:
        return pickle.load(f)


def gpt_answer(question):
    index = search_index()
    # TODO: use the links in the document metadata to extract related
    # docs
    result = CHAIN(
        {
            "input_documents": index.similarity_search(question, k=4),
            "question": question,
        },
        return_only_outputs=True,
    )

    return result["output_text"]


# def gpt3():
#     response = openai.Completion.create(
#         prompt=prompt + start_text,
#         engine=engine,
#         max_tokens=response_length,
#         temperature=temperature,
#         top_p=top_p,
#         frequency_penalty=frequency_penalty,
#         presence_penalty=presence_penalty,
#         stop=stop_seq,
#     )
#     answer = response.choices[0]['text']
#     new_prompt = prompt + start_text + answer + restart_text
#     return answer, new_prompt


def chat():
    while True:
        prompt = input(' ')
        answer = gpt_answer(prompt)
        print(answer)


if __name__ == '__main__':
    args = sys.argv
    if len(args) == 1:
        chat()
    if len(args) == 3:
        command = sys.argv[1]
        if command == "index":
            path = sys.argv[2]
            build_search_index_and_embeddings(path)
            print("Indexing complete!")
        else:
            print(f"Unknown command \"{command}\"")
    else:
        print("Too many arguments")
